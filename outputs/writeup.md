# Lab 25 Write-up — GPU FinOps Optimization

## 1. Baseline vs. Optimized

NimbusAI đang trả chi phí GPU baseline khoảng **$27,133/tháng**. Sau các tối ưu trong lab, chi phí optimized còn **$14,626/tháng**, tiết kiệm **$12,507/tháng**, tương đương **46%**.

Ở metric quan trọng nhất là `$/1M-token`, inference giảm từ **$6.488/1M-token** xuống **$1.126/1M-token**. Đây là metric tốt hơn `$/GPU-hr` vì nó đo chi phí trên đơn vị giá trị thực sự được phục vụ cho người dùng.

## 2. Phân tích từng đòn bẩy

| Lever | Savings / month | Nhận xét |
|---|---:|---|
| Purchasing (spot/reserved) | $10,040 | Đóng góp lớn nhất; workload interruptible chuyển sang spot, workload duty-cycle cao dùng reserved. |
| Inference (cascade/cache/batch) | $1,212 | Cascade route request dễ sang model nhỏ, prompt cache giảm input cost, batch API giảm traffic không cần realtime. |
| Right-size util-lies | $655 | GPU có util cao nhưng MFU thấp nên nên hạ tier hoặc đổi GPU phù hợp workload hơn. |
| Kill idle GPUs | $600 | Tắt GPU idle thay vì trả tiền cả ngày. |

Ưu tiên triển khai nên là purchasing trước vì ROI lớn nhất, sau đó inference routing/cache/batch vì tác động trực tiếp đến `$/1M-token`, rồi xử lý GPU util-lie và idle capacity.

## 3. GPU-Util Lie

M1 phát hiện **gpu-h100-4** và **gpu-a10g-1** có GPU-Util cao nhưng MFU dưới ngưỡng 30%. Trường hợp chính là `gpu-h100-4`: GPU-Util khoảng 98% nhưng MFU chỉ khoảng 0.20, nghĩa là GPU trông có vẻ bận nhưng chỉ dùng được khoảng 1/5 FLOPs peak.

Điều này thường xảy ra khi workload bị memory stall, kernel launch overhead, hoặc batch/shape không tận dụng tốt tensor cores. Nếu chỉ nhìn `nvidia-smi` GPU-Util thì dễ tưởng GPU đã được dùng hiệu quả, trong khi chi phí tính theo GPU-hour vẫn bị tính đầy đủ.

Idle waste hiện tại là **$20/ngày**, tương đương **$600/tháng**.

## 4. Extension 1 — Cache Economics

Đã thêm `cache_break_even_reads()` và `cache_is_worth_it()` trong `finops/pricing.py`, sau đó áp dụng vào M2. Cache chỉ được tính khi nhóm prefix vượt ngưỡng hòa vốn.

Giả định chi phí ghi cache tương đương 1 lần input và cached read còn 10% giá input. Mỗi lần đọc cache tiết kiệm 90%, nên break-even là:

```text
1.0 / (1 - 0.10) = 1.11 reads
```

Dataset hiện tại có trung bình **150.0 reads/prefix**, và **16/16 prefix groups** vượt ngưỡng. Vì vậy prompt caching là có lợi trong dữ liệu lab này và được bật cho toàn bộ nhóm prefix có cached tokens.

Insight chính: caching chỉ nên bật khi prefix thực sự được tái sử dụng. Nếu prefix chỉ dùng một lần, cache có thể làm tăng chi phí do phải trả chi phí ghi/lưu cache.

## 5. Extension 2 — Reasoning Budget

Đã tách riêng traffic `is_reasoning=1` trong M2 và đưa kết quả vào M5 report.

Kết quả đo được:

| Metric | Value |
|---|---:|
| Reasoning traffic | 8.4% requests |
| Share of optimized cost | 16.5% |
| Share of inference energy | 94.0% Wh |
| 10% cap | Đã đạt, không tiết kiệm thêm |
| 5% default cap | Tiết kiệm khoảng $12/tháng và 357,972 Wh/tháng |

Reasoning tốn năng lượng rất lớn vì lab dùng multiplier khoảng 80x cho reasoning queries. Dù chỉ chiếm 8.4% traffic, nó chiếm 94.0% năng lượng inference.

Policy đề xuất: chỉ bật reasoning cho eval hoặc request có độ phức tạp cao; traffic mặc định nên cap ở 5%, các request còn lại route qua model thường hoặc small model trước.

## 6. Sustainability

Snapshot hiện tại:

- Energy/query: **0.24 Wh**
- Carbon/query tại `us-east-1`: **0.091 gCO2e**
- Cheapest+cleanest region: **europe-north1**

Với workload interruptible hoặc batch, nên ưu tiên chuyển sang region sạch/rẻ khi latency không phải ràng buộc chính. Điều này giảm cả carbon và chi phí điện.

## 7. Khuyến nghị cho NimbusAI

1. Áp dụng policy purchasing: spot cho job interruptible có checkpoint, reserved cho workload chạy ổn định nhiều giờ/ngày.
2. Đưa inference gateway vào production với cascade, prompt cache có break-even check, và batch API cho eval/nightly traffic.
3. Thêm guardrail cho reasoning: default cap 5%, chỉ escalates sang reasoning khi confidence thấp hoặc task complexity cao.
4. Theo dõi MFU/MBU thay vì chỉ GPU-Util để phát hiện GPU đang bận nhưng không hiệu quả.
5. Tắt GPU idle và ưu tiên region sạch/rẻ cho workload có thể dời lịch.
