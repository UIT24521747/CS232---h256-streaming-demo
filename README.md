# H256: Demo pipeline video end-to-end (CS232)

Một demo nhỏ, viết tay hoàn toàn, mô phỏng ý tưởng cốt lõi của streaming video hiện đại:

1. tách video thành các frame I/P/B,
2. nén bằng một codec đồ chơi tự viết ("H256"),
3. phát qua HTTP thật thông qua một load balancer round-robin,
4. **nhiều người dùng (mỗi người một `user_id`) cùng xem chung một video**, mỗi người có buffer, băng thông và mức chất lượng riêng,
5. giải mã phía client và phóng to bằng một bước super-resolution đơn giản.

Mọi bước đều đủ nhỏ để đọc hết và tính tay cho một frame cụ thể của một user cụ thể.

> **H256 không phải H.265.** Tên gọi chỉ là một cái nháy mắt: cùng ý tưởng (loại frame, dự đoán, phần dư, lượng tử hóa, mã hóa entropy, bitstream) nhưng không có phần phức tạp thật (không DCT, không CABAC, không bộ lọc in-loop). Không có lời gọi thư viện nào giả làm codec: không FFmpeg, không GAN, không mô hình pretrained, không Kafka/K8s/microservices.

---

## Mục lục

1. [Demo này thể hiện điều gì](#1-demo-này-thể-hiện-điều-gì)
2. [Chạy nhanh](#2-chạy-nhanh)
3. [Nhập video đầu vào](#3-nhập-video-đầu-vào)
4. [Kiến trúc](#4-kiến-trúc)
5. [Cơ sở toán học](#5-cơ-sở-toán-học)
6. [Nhiều người dùng: session, buffer và MV theo user](#6-nhiều-người-dùng-session-buffer-và-mv-theo-user)
7. [Ví dụ một datapoint được lưu](#7-ví-dụ-một-datapoint-được-lưu)
8. [Truy vết một frame từ đầu đến cuối](#8-truy-vết-một-frame-từ-đầu-đến-cuối)
9. [Web dashboard](#9-web-dashboard)
10. [Cấu trúc thư mục](#10-cấu-trúc-thư-mục)
11. [Luồng thực thi tuần tự](#11-luồng-thực-thi-tuần-tự)
12. [Kiểm thử](#12-kiểm-thử)
13. [Vì sao các con số trông như vậy](#13-vì-sao-các-con-số-trông-như-vậy)
14. [Demo này *không* phải là gì](#14-demo-này-không-phải-là-gì)

---

## 1. Demo này thể hiện điều gì

1. Mã hóa video I/P/B (mẫu GOP cố định + vector chuyển động (MV) tìm bằng block matching thật)
2. Codec H256 đơn giản (dự đoán → phần dư → lượng tử hóa → mã hóa entropy)
3. Bitstream đúng từng byte, đọc được bằng mắt
4. **Nhập video tùy ý** (CLI hoặc upload trên dashboard)
5. Streaming HTTP (hai node Flask thật)
6. Load balancing round-robin với log request hiển thị được
7. **Nhiều user đồng thời**, mỗi user có `user_id`, buffer, băng thông và lịch sử frame riêng
8. Adaptive bitrate (ABR) phía client + giải mã
9. Super-resolution đơn giản (bilinear + sharpen viết tay, không mô hình)
10. Truy vết end-to-end một frame cụ thể của một user cụ thể

## 2. Chạy nhanh

```bash
pip install -r requirements.txt

python demo.py                          # chạy toàn bộ pipeline một lần, in bảng tóm tắt
python demo.py --video path/to/clip.mp4 # dùng video của bạn thay cho clip mẫu
python demo.py --users 4                # mô phỏng 4 user cùng xem một video
python demo.py --users 4 --user u2      # in danh sách frame theo thứ tự gửi tới user u2
python demo.py --frame 12 --user u2     # in trace 13 bước cho frame 12 của user u2
python demo.py --bandwidth 800          # băng thông mặc định cho mọi user (kbps)
python demo.py --qp 16                  # lượng tử hóa thô hơn -> bitstream nhỏ hơn, mất mát nhiều hơn
python demo.py --web                    # mở dashboard trên trình duyệt
```

Kiểm tra bitstream thô bằng mắt:

```bash
python tools/inspect_bitstream.py outputs/<video_id>/segments/540p/segment_000.bin
python tools/inspect_bitstream.py outputs/<video_id>/segments/540p/segment_000.bin --frame 1 --block 5
```

## 3. Nhập video đầu vào

Có ba cách đưa video vào pipeline, và cả ba đi qua **cùng một đường code**
(`pipeline.read_video_frames()` gọi `cv2.VideoCapture`, sau đó resize về thang 240p/360p/540p):

| Cách | Lệnh / thao tác | Ghi chú |
|---|---|---|
| Clip mẫu có sẵn | `python demo.py` | Dùng `data/sample.mp4`; nếu thiếu, tự sinh lại |
| Video từ CLI | `python demo.py --video path/to/clip.mp4` | Mọi định dạng OpenCV đọc được |
| Upload trên web | Nút **"Tải video lên"** trên dashboard | `POST /api/upload`, lưu vào `data/uploads/<video_id>.mp4` |

Mỗi video được gán một `video_id` (hash SHA-1 rút gọn của nội dung file), nên cùng một file upload hai lần sẽ dùng lại kết quả encode đã có trong `outputs/<video_id>/`.

**Giới hạn đầu vào khuyến nghị:** tối đa ~10 giây và tự động cắt về tối đa `--max-frames` (mặc định 64 frame), vì motion search vét cạn chạy trên CPU và tăng tuyến tính theo số frame.

### Clip mẫu `data/sample.mp4`

- **Bộ sinh:** `data/generate_sample.py`, chỉ dùng NumPy + OpenCV (`cv2.VideoWriter`), không tải gì về.
- **Thông số:** 256×144, 12 fps, 32 frame (~2,7 s), kênh màu BGR.
- **Nội dung:** nền gradient hai tông trượt ngang, một quả bóng trắng trôi và nảy, một hình chữ nhật xanh lá đứng yên ở góc. Chuyển động được thiết kế để **nằm trong cửa sổ tìm kiếm ±8 px** của encoder, nhờ vậy P/B frame thực sự có ý nghĩa.

```bash
python data/generate_sample.py --frames 48 --fps 24 --out data/sample.mp4
```

Với video thật có chuyển động nhanh hơn ±8 px giữa hai anchor, phần dư P/B sẽ lớn lên. Đó là đánh đổi bình thường giữa phạm vi tìm kiếm và chi phí tính toán, không phải lỗi.

## 4. Kiến trúc

```
VIDEO ĐẦU VÀO (sample.mp4 | --video | upload)
      |
      v
+---------------+   codec/h256_encoder.py: classify_frames()
|  PHÂN LOẠI    |   GOP cố định 8 frame: 0=I, lẻ=P, chẵn=B
|  I / P / B    |   (sắp xếp lại: anchor trước, B sau)
+-------+-------+
        v
+---------------+   predictor -> residual -> quantizer -> entropy
|  H256 ENCODER |   encode MỘT LẦN cho mỗi mức chất lượng (240p/360p/540p),
+-------+-------+   dùng chung cho mọi user
        v
   BITSTREAM + CHỈ MỤC MV
   outputs/<video_id>/segments/<quality>/segment_XXX.bin
   outputs/<video_id>/mv_index/<quality>.json
        |
        v
+---------------+   streaming/load_balancer.py: counter mod N
|  LOAD         |   log: user=... request=... -> node[i] (Round-Robin)
|  BALANCER     |
+-------+-------+
        v
+---------------+   streaming/server.py: 2 node Flask giống hệt nhau
|  STREAMING    |   GET /segment/<video_id>/<quality>/<name>?user=<user_id>
|  NODES        |
+-------+-------+
        v
+-----------------------------------------------+
|  SESSION MANAGER  (streaming/session.py)       |
|   u1: bandwidth, quality, buffer[], history[]  |
|   u2: ...                                      |
|   u3: ...                                      |
+-------+---------------------------------------+
        v  (mỗi user một StreamingClient riêng, chạy song song)
+---------------+   codec/h256_decoder.py: đọc MV trực tiếp (không tìm kiếm),
|  H256 DECODER |   giải lượng tử, bù chuyển động / dự đoán DC, tái tạo
+-------+-------+
        v
+---------------+   sr/simple_sr.py: bilinear x2 + kernel sharpen 3x3
|  SIMPLE SR    |
+-------+-------+
        v
   FRAME CUỐI     outputs/<video_id>/users/<user_id>/frame_<id>/05_sr.png
```

## 5. Cơ sở toán học

Ký hiệu: frame gốc $F_t$, frame tái tạo (sau giải mã) $\tilde F_t$, frame dự đoán $\hat F_t$, kích thước khối $B = 16$, tham số lượng tử $QP$.

### 5.1. Phân loại frame và thứ tự giải mã

Mỗi segment gồm 8 frame là một mini-GOP độc lập. Với chỉ số cục bộ $k = t \bmod 8$:

$$
\text{type}(t) =
\begin{cases}
I & k = 0 \\
P & k \text{ lẻ}, \quad \text{ref} = \text{anchor liền trước} \\
B & k \text{ chẵn}, k \neq 0, \quad \text{ref} = (t-1,\ t+1)
\end{cases}
$$

Vì B cần frame *tương lai* đã được tái tạo, thứ tự **gửi/giải mã** khác thứ tự **hiển thị**:

| Thứ tự hiển thị (cục bộ) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Loại | I | P | B | P | B | P | B | P |
| Tham chiếu | – | 0 | 1, 3 | 1 | 3, 5 | 3 | 5, 7 | 5 |

Thứ tự giải mã: $0, 1, 3, 5, 7, 2, 4, 6$. Đây cũng chính là thứ tự frame hiện ra trong danh sách "frame đã nhận" của từng user.

### 5.2. Chia khối

$$
N_{\text{blocks}} = \left\lceil \frac{W}{B} \right\rceil \cdot \left\lceil \frac{H}{B} \right\rceil
\qquad \text{ví dụ } 160\times 96 \Rightarrow 10 \cdot 6 = 60 \text{ khối}
$$

### 5.3. Dự đoán intra (I-frame): DC

Với khối $b$ có khối trái $L$ và khối trên $T$ đã được tái tạo:

$$
\hat F_b =
\begin{cases}
\operatorname{round}\!\left(\dfrac{\overline{\tilde F_L} + \overline{\tilde F_T}}{2}\right) & \text{có cả } L, T \\[4pt]
\overline{\tilde F_L} \ \text{hoặc}\ \overline{\tilde F_T} & \text{chỉ có một} \\[4pt]
128 & \text{khối góc trên-trái}
\end{cases}
$$

### 5.4. Ước lượng chuyển động (P/B-frame)

Tìm vét cạn trong cửa sổ $S = 8$:

$$
(d_x^*, d_y^*) = \arg\min_{|d_x|,|d_y| \le S} \ \frac{1}{3B^2}\sum_{c}\sum_{(x,y)\in b} \Big(F_t^c(x,y) - \tilde F_{\text{ref}}^c(x+d_x,\ y+d_y)\Big)^2
$$

Số vị trí thử cho mỗi khối: $(2S+1)^2 = 289$.

**P-frame:**
$$
\hat F_t(x,y) = \tilde F_{\text{ref}}(x + d_x,\ y + d_y)
$$

**B-frame** (hai MV độc lập, một cho mỗi tham chiếu):
$$
\hat F_t(x,y) = \left\lfloor \frac{\tilde F_{t-1}(x+d_x^{(0)},\ y+d_y^{(0)}) + \tilde F_{t+1}(x+d_x^{(1)},\ y+d_y^{(1)}) + 1}{2} \right\rfloor
$$

### 5.5. Phần dư và lượng tử hóa

$$
R_t = F_t - \hat F_t
$$

$$
L = \operatorname{round}\!\left(\frac{R_t}{QP}\right), \qquad
R'_t = L \cdot QP, \qquad
\left|R_t - R'_t\right| \le \frac{QP}{2}
$$

### 5.6. Tái tạo (vòng kín)

$$
\tilde F_t = \operatorname{clip}_{[0,255]}\big(\hat F_t + R'_t\big)
$$

Encoder lưu $\tilde F_t$ (không phải $F_t$) vào DPB để làm tham chiếu, nên encoder và decoder luôn thấy đúng cùng một tham chiếu và sai số không tích lũy dọc GOP.

### 5.7. Mã hóa entropy: RLE số 0

Chuỗi mức lượng tử $(\ell_1, \ell_2, \dots)$ được biểu diễn thành các cặp $(\text{run}, \text{value})$, trong đó `run` là số số 0 đứng trước `value`:

$$
[0,0,0,1,0,0,-2,0,0,0] \ \longrightarrow\ (3, 1),\ (2, -2),\ \text{EOB}
$$

### 5.8. Kích thước và tỷ lệ nén

$$
\text{Raw} = W \cdot H \cdot 3 \ \text{(byte)}, \qquad
\text{Compression} = \left(1 - \frac{\text{Encoded}}{\text{Raw}}\right)\times 100\%
$$

Ví dụ: $160 \cdot 96 \cdot 3 = 46080$ byte, encode còn $5715$ byte $\Rightarrow$ nén $87{,}6\%$.

Bitrate của một segment dài $T_{\text{seg}} = 8 / \text{fps}$ giây:

$$
\text{bitrate}_{\text{kbps}} = \frac{8 \cdot \text{size}_{\text{byte}}}{1000 \cdot T_{\text{seg}}}
$$

### 5.9. Độ đo chất lượng

$$
\text{MSE} = \frac{1}{3WH}\sum_{c,x,y}\big(F^c(x,y) - \tilde F^c(x,y)\big)^2,
\qquad
\text{PSNR} = 10 \log_{10}\frac{255^2}{\text{MSE}} \ \text{(dB)}
$$

$$
\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)},
\quad C_1 = (0{,}01\cdot 255)^2,\ C_2 = (0{,}03 \cdot 255)^2
$$

PSNR/SSIM chỉ được tính giữa frame tái tạo và frame gốc **ở cùng độ phân giải**, tức là đo cái giá của riêng codec.

### 5.10. Load balancing round-robin

Với bộ đếm toàn cục $c$ (tăng sau mỗi request, bất kể user nào gửi) và $N$ node:

$$
\text{node}(c) = c \bmod N
$$

Vì bộ đếm dùng chung, các request của nhiều user đan xen nhau, nên một user có thể nhận hai segment liên tiếp từ cùng một node. Dashboard hiển thị đúng điều này.

### 5.11. Adaptive bitrate (ABR)

Với thang chất lượng $\mathcal Q = \{240p, 360p, 540p\}$, bitrate danh định $r_q$ và hệ số an toàn $\alpha = 0{,}8$:

$$
q^*(u) = \max\{\, q \in \mathcal Q \ :\ r_q \le \alpha \cdot \text{bw}(u) \,\}
\quad (\text{nếu không có, chọn } 240p)
$$

Thời gian chờ mô phỏng mạng chậm cho user $u$:

$$
t_{\text{target}} = \frac{8 \cdot \text{size}}{1000 \cdot \text{bw}(u)}, \qquad
t_{\text{sleep}} = \max\big(0,\ t_{\text{target}} - t_{\text{thực}}\big)
$$

### 5.12. Super-resolution đơn giản

**Bilinear:** với tọa độ nguồn $(x', y') = \big(\tfrac{x}{s}, \tfrac{y}{s}\big)$, $x_0 = \lfloor x' \rfloor$, $a = x' - x_0$, $b = y' - y_0$:

$$
I(x,y) = (1-a)(1-b)\,I_{00} + a(1-b)\,I_{10} + (1-a)\,b\,I_{01} + a\,b\,I_{11}
$$

**Sharpen (unsharp mask):**

$$
I_{\text{sharp}} = \operatorname{clip}_{[0,255]}\big(I + \lambda\,(I - G * I)\big)
$$

với $G$ là kernel làm mờ 3×3 và $\lambda$ là độ mạnh. Khi $G$ là trung bình 4 lân cận và $\lambda = 1$, ta được kernel quen thuộc:

$$
K = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}
$$

> Giá trị $G$, $\lambda$ cụ thể lấy theo `sr/simple_sr.py`.

## 6. Nhiều người dùng: session, buffer và MV theo user

### 6.1. Ý tưởng

- Có **một** video đầu vào; mọi user cùng xem video đó.
- Mỗi user có một `user_id` riêng (`u1`, `u2`, ...) và một **session** trong `SessionManager`, gồm: băng thông, mức chất lượng hiện tại, buffer, và lịch sử các frame đã nhận.
- Mỗi user chạy một `StreamingClient` riêng trong một thread riêng, cùng gửi request qua **một** load balancer.
- Chọn một user bất kỳ để xem **thứ tự các frame được gửi về user đó**, mỗi frame kèm: loại (I/P/B), tham chiếu, MV, kích thước, PSNR, node phục vụ, mức chất lượng, thời điểm nhận.

### 6.2. MV được lấy theo user như thế nào

Cần nói rõ một điểm: **MV là thuộc tính của nội dung video đã được encode, không phải của user.** Hai user cùng xem frame 12 ở cùng mức 360p sẽ nhận cùng một bitstream và cùng một bộ MV. Việc encode riêng cho từng user vừa tốn CPU gấp nhiều lần, vừa không đúng với cách streaming thật vận hành (CDN chỉ phục vụ file đã encode sẵn).

Vì vậy `user_id` không dùng để *tính* MV, mà dùng để *tra* MV thông qua trạng thái session:

```
user_id ──► session[user_id]
              │
              ├─ quality đã chọn cho segment chứa frame  (do ABR quyết định, khác nhau giữa các user)
              ▼
         khóa (video_id, quality, frame_id)
              │
              ▼
         mv_index/<quality>.json  ──►  MV của từng khối
```

$$
\text{MV}(u, t) = \text{MVIndex}\big[\text{video\_id},\ q_u(\text{seg}(t)),\ t\big],
\qquad \text{seg}(t) = \left\lfloor \frac{t}{8} \right\rfloor
$$

Nhờ vậy MV *thực sự khác nhau giữa các user* khi họ có băng thông khác nhau: user băng thông thấp nhận segment 240p (khối 16×16 phủ vùng ảnh lớn hơn, MV có biên độ nhỏ hơn), user băng thông cao nhận 540p. Nếu băng thông của một user thay đổi giữa chừng, các segment sau của user đó sẽ đổi sang bộ MV của mức chất lượng mới, và danh sách frame hiển thị rõ điểm chuyển này.

### 6.3. Buffer

Mỗi session giữ một hàng đợi các frame đã giải mã, kèm hai số đo:

$$
\text{buffer\_level}(u) = \frac{\#\text{frame trong buffer}}{\text{fps}} \ \text{(giây)},
\qquad
\text{rebuffer}(u) \iff \text{buffer\_level}(u) = 0 \text{ khi đang phát}
$$

## 7. Ví dụ một datapoint được lưu

### 7.1. Một bản ghi frame trong lịch sử của user

Mỗi frame user nhận được sinh ra một bản ghi như sau, lưu trong
`session.history` và ghi ra `outputs/<video_id>/users/<user_id>/history.jsonl`:

```json
{
  "user_id": "u2",
  "video_id": "a3f9c1e2",
  "recv_order": 13,
  "frame_id": 12,
  "segment": "segment_001.bin",
  "display_index_in_gop": 4,
  "decode_index_in_gop": 6,
  "type": "B",
  "refs": [11, 13],
  "quality": "360p",
  "resolution": [160, 96],
  "qp": 8,
  "node": "Node-1",
  "lb_counter": 27,
  "bandwidth_kbps": 1200,
  "raw_bytes": 46080,
  "encoded_bytes": 5715,
  "compression_pct": 87.6,
  "residual_range": [-71, 66],
  "residual_mse": 19.41,
  "psnr_db": 41.78,
  "ssim": 0.981,
  "n_blocks": 60,
  "mv_ref": "mv_index/360p.json#frame=12",
  "recv_time_ms": 2143,
  "buffer_level_s": 1.25
}
```

`recv_order = 13` vì frame 12 là frame thứ 7 theo thứ tự giải mã của segment 1 ($8 + 6 = 14$ frame đã tới, đánh số từ 0 là 13). MV không lưu thẳng trong bản ghi mà trỏ tới chỉ mục MV dùng chung, để không nhân bản dữ liệu theo số user.

### 7.2. Một khối trong chỉ mục MV

```json
{
  "frame_id": 12,
  "block": 5,
  "pos": [80, 0],
  "mv": [[-2, 1], [2, -1]],
  "match_mse": [14.2, 16.8],
  "nonzero_levels": 7,
  "rle_bytes": 23
}
```

Với B-frame, `mv[0]` trỏ về frame 11 và `mv[1]` trỏ về frame 13. Khối 5 nằm ở hàng 0, cột 5, nên góc trên-trái là $(5 \cdot 16,\ 0) = (80, 0)$.

### 7.3. Tính tay một pixel của khối đó

Lấy pixel $(85, 3)$, kênh G, với $QP = 8$:

| Bước | Công thức | Giá trị |
|---|---|---|
| Gốc | $F_{12}(85,3)$ | 140 |
| Tham chiếu trước | $\tilde F_{11}(85-2,\ 3+1) = \tilde F_{11}(83, 4)$ | 131 |
| Tham chiếu sau | $\tilde F_{13}(85+2,\ 3-1) = \tilde F_{13}(87, 2)$ | 137 |
| Dự đoán | $\lfloor (131 + 137 + 1)/2 \rfloor$ | 134 |
| Phần dư | $140 - 134$ | 6 |
| Mức lượng tử | $\operatorname{round}(6/8)$ | 1 |
| Giải lượng tử | $1 \cdot 8$ | 8 |
| Tái tạo | $\operatorname{clip}(134 + 8)$ | 142 |
| Sai số | $\lvert 140 - 142 \rvert \le QP/2 = 4$ | 2 ✓ |

### 7.4. Bố cục byte của frame trong bitstream (ví dụ)

```
Frame header (little-endian)
  u16 frame_id      = 0x000C        (12)
  u8  frame_type    = 0x02          (0=I, 1=P, 2=B)
  i16 ref1          = 0x000B        (11)
  i16 ref2          = 0x000D        (13)
  u8  qp            = 0x08
  u16 width         = 0x00A0        (160)
  u16 height        = 0x0060        (96)
  u32 payload_bytes = ...

Block header (lặp lại 60 lần)
  i8 dx0, i8 dy0, i8 dx1, i8 dy1    = FE 01 02 FF   (-2, 1, 2, -1)
  u16 n_pairs[3]                    (số cặp RLE của từng kênh B, G, R)

Payload RLE: các cặp (u8 run, i8 value)
  03 01   02 FE   ...              (3, 1), (2, -2), ...
```

> Tên và kích thước trường ở trên là ví dụ minh họa; nguồn chân lý là `codec/bitstream.py`, và `tools/inspect_bitstream.py` in đúng bố cục thực tế.

## 8. Truy vết một frame từ đầu đến cuối

```bash
python demo.py --users 3 --frame 12 --user u2
```

```
==================================================
TRUY VẾT END-TO-END  (user u2)
==================================================

[1] NGUỒN
Frame ID            : 12
Độ phân giải        : 256x144
Kích thước gốc      : 110592 byte

[2] LOẠI FRAME
Loại                : B
Tham chiếu          : 11, 13
Thứ tự nhận (u2)    : 13

[3] DỰ ĐOÁN
Phương pháp         : trung bình bù chuyển động từ frame 11 và 13

[4] PHẦN DƯ
Khoảng giá trị      : [-71, 66]
MSE phần dư         : 19.41

[5] LƯỢNG TỬ HÓA
QP                  : 8

[6] H256 ENCODE
Kích thước thô      : 46080 byte
Sau khi encode      : 5715 byte
Tỷ lệ nén           : 87.6%

[7] BITSTREAM
Chất lượng          : 360p   (ABR của u2 @ 1200 kbps)
Segment             : segment_001.bin
Chỉ mục MV          : mv_index/360p.json#frame=12

[8] LOAD BALANCER
Request             : u2 -> 360p/segment_001.bin
Node được chọn      : Node-1   (counter = 27)
Chính sách          : Round-Robin

[9] CLIENT
Số byte nhận        : 5715
Buffer của u2       : 1.25 s

[10] H256 DECODE
Loại frame          : B
Tham chiếu          : 11, 13

[11] TÁI TẠO
Độ phân giải        : 160x96
PSNR tái tạo        : 41.78 dB

[12] SUPER RESOLUTION
Đầu vào             : 160x96
Đầu ra              : 320x192
Phương pháp         : Bilinear x2 + kernel sharpen viết tay

[13] KẾT QUẢ
Đầu ra              : outputs/a3f9c1e2/users/u2/frame_12/05_sr.png

==================================================
HOÀN TẤT
==================================================
```

Lệnh này đồng thời ghi ra `outputs/<video_id>/users/u2/frame_12/`:

```
01_original.png
02_prediction.png
03_residual.png
04_reconstructed.png
05_sr.png
mv_overlay.png    <- MV của từng khối vẽ thành mũi tên
trace.json
summary.png       <- một ảnh gồm cả 5 bước + các con số chính
```

## 9. Web dashboard

```bash
python demo.py --web      # http://127.0.0.1:7000/
```

Một trang Flask duy nhất (không build step, không framework JS).

### 9.1. Màn hình tổng quan

- **Tải video lên** hoặc dùng clip mẫu; hiển thị `video_id`, độ phân giải, số frame, fps.
- **Thêm / xóa user**; mỗi user có thanh trượt băng thông riêng.
- **Lưới user:** mỗi ô là một user, phát video *đúng như user đó đang nhận*, gồm:
  - khung video đã giải mã (và bản SR) ở mức chất lượng của user,
  - nhãn frame hiện tại (`#12 · B · 360p`),
  - node đang phục vụ,
  - thanh buffer và cờ rebuffer,
  - PSNR trung bình và tổng byte đã nhận.
- **Log load balancer** chung cho mọi user (`u1 -> Node-2`, `u3 -> Node-1`, ...) để thấy rõ round-robin đan xen.
- **Thanh trượt QP**, áp dụng cho lần encode tiếp theo.

Khi kéo băng thông của một user trong lúc đang chạy, chỉ ô của user đó đổi chất lượng từ segment kế tiếp; các user khác không bị ảnh hưởng.

### 9.2. Màn hình chi tiết user

Nhấn vào một ô user để mở bảng **frame theo thứ tự nhận**:

| # nhận | Frame | Loại | Ref | Chất lượng | Node | Kích thước | Nén | PSNR | Thời điểm |
|---:|---:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|
| 8 | 8 | I | – | 360p | Node-2 | 31 204 B | 32,3% | 43,10 dB | 1 610 ms |
| 9 | 9 | P | 8 | 360p | Node-2 | 7 880 B | 82,9% | 42,05 dB | 1 610 ms |
| ... | | | | | | | | | |
| 13 | 12 | B | 11, 13 | 360p | Node-1 | 5 715 B | 87,6% | 41,78 dB | 2 143 ms |

*(Các số trong bảng minh họa, dashboard hiển thị số của lần chạy thật.)*

Nhấn vào một hàng để xem ảnh gốc / dự đoán / phần dư / tái tạo / SR, **lớp phủ MV** của frame đó, và trace 13 bước.

### 9.3. API

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| `POST` | `/api/upload` | Upload video, trả về `video_id` |
| `POST` | `/api/users` | Tạo user mới `{bandwidth}` → `user_id` |
| `DELETE` | `/api/users/<user_id>` | Xóa user |
| `POST` | `/api/users/<user_id>/bandwidth` | Đổi băng thông ngay lập tức |
| `GET` | `/api/users` | Trạng thái tóm tắt của mọi user (cho màn hình tổng quan) |
| `GET` | `/api/users/<user_id>/frames` | Lịch sử frame theo thứ tự nhận |
| `GET` | `/api/users/<user_id>/frames/<frame_id>` | Chi tiết + MV + trace của một frame |
| `POST` | `/api/run` | Bắt đầu phát `{video_id, qp}` cho mọi user |
| `GET` | `/api/state` | Trạng thái chung (log LB, tiến độ) |
| `GET` | `/outputs/<path>` | Phục vụ ảnh trace |

Dashboard gọi **đúng các hàm** trong `pipeline.py` như CLI; nó khởi động hai node Flask thật cùng load balancer trong các thread nền và giao tiếp qua HTTP thật.

## 10. Cấu trúc thư mục

```
demo.py                 điểm vào duy nhất (CLI + in trace)
pipeline.py             điều phối: nối codec / streaming / sr
codec/
  frame.py              Frame, FrameType, tiện ích lưới khối
  predictor.py          dự đoán I/P/B (DC intra + block matching)
  residual.py           residual = original - prediction
  quantizer.py          lượng tử hóa vô hướng đều
  entropy.py            RLE số 0
  bitstream.py          bố cục byte bằng struct
  mv_index.py           ghi/đọc chỉ mục MV theo (video_id, quality, frame_id)
  h256_encoder.py       classify_frames / encode_order / H256Encoder
  h256_decoder.py       H256Decoder, decode_segment
streaming/
  server.py             một node streaming (Flask)
  load_balancer.py      load balancer round-robin (Flask)
  client.py             client ABR (một instance cho mỗi user)
  session.py            SessionManager, UserSession, buffer, history
sr/simple_sr.py         resize bilinear + sharpen
metrics/metrics.py      MSE / PSNR / SSIM viết tay
tools/
  inspect_bitstream.py  in bitstream dạng đọc được
  visualize_trace.py    ảnh từng bước, ảnh ghép, lớp phủ MV
web/                    dashboard (Flask + JS thuần)
data/
  generate_sample.py    sinh clip tổng hợp
  uploads/              video được upload
tests/                  bộ pytest
legacy/                 hai notebook brainstorm ban đầu (Video-SR / GAN,
                        giữ để tham khảo, không thuộc pipeline này)
```

## 11. Luồng thực thi tuần tự

### A. `python demo.py --users N`

```
demo.py : main() -> run_pipeline(args)

  [1/7] Chuẩn bị video
        demo.py : _ensure_video()
          -> data/generate_sample.py : generate()     (nếu thiếu sample.mp4)
        pipeline.py : video_id_of(path)               (SHA-1 rút gọn)

  [2/7] Đọc và tạo thang chất lượng
        pipeline.py : load_source_video()  -> read_video_frames() -> cv2.VideoCapture
                                            (cắt về tối đa --max-frames)
        pipeline.py : build_quality_tiers()
          -> sr/simple_sr.py : resize_bilinear()      (x3 mức)

  [3/7] Encode / reuse cache
        pipeline.py : ensure_encoded()
          nếu outputs/<video_id>/manifest.json đã khớp qp + num_frames -> cache hit,
          decode lại segment có sẵn để dựng FrameStats (không cần motion search)
          ngược lại, với mỗi mức, với mỗi chunk 8 frame:
            -> classify_frames() -> encode_order()
            -> H256Encoder.encode_frame()
                 I: predict_intra_block()
                 P: predict_inter() -> motion_search()
                 B: predict_bidirectional() -> motion_search() x2
                 -> compute_residual() -> quantize() / dequantize()
                 -> reconstruct()  (vào DPB)
                 -> _pack() -> bitstream.py / entropy.py
            -> codec/mv_index.py : frame_entries() -> write_index()
          -> ghi segments/*.bin, mv_index/<quality>.json, manifest.json

  [4/7] Khởi động hạ tầng
        pipeline.py : start_streaming()
          -> server.py : create_app() x2          Node-1 (:6101), Node-2 (:6102)
          -> load_balancer.py : create_app()      LB (:6100)

  [5/7] Tạo user
        streaming/session.py : SessionManager.create_user() x N
          -> mỗi user: UserSession(user_id, bandwidth, buffer=[], history=[])

  [6/7] Phát song song
        pipeline.py : run_all_users()   (một thread cho mỗi user)
          -> client.py : StreamingClient(user_id).fetch_manifest()
          với mỗi segment:
            -> choose_quality(session.bandwidth)
            -> HTTP GET :6100/segment/<video_id>/<quality>/<name>?user=<id>
                 -> load_balancer.py : pick_node()  (counter dùng chung)
                 -> server.py : segment()
            -> time.sleep(t_sleep)
            -> h256_decoder.py : decode_segment()   (theo thứ tự giải mã)
            -> với mỗi frame giải mã:
                 -> mv_index.lookup(video_id, quality, frame_id)
                 -> metrics.py : psnr() / ssim()
                 -> session.py : UserSession.push(record)   (buffer + history)

  [7/7] Báo cáo
        demo.py : print_user_summary()                (mọi user)
        demo.py : print_frame_table(user=args.user)    (nếu có --user)
        handles.shutdown()

Super-resolution (sr/simple_sr.py : simple_super_resolution()) không chạy
hàng loạt cho mọi frame ở bước này -- nó chỉ chạy khi thực sự cần hiển thị
ảnh: cho đúng frame đang được trace (--frame/--user, hoặc khi dashboard
web nhận một frame mới, xem luồng C bên dưới). Tính SR cho mọi frame của
mọi user mà không ai xem là lãng phí CPU vô ích.
```

### B. `python demo.py --frame 12 --user u2`: toàn bộ A, sau đó

```
pipeline.py : trace_user_frame(video_id, session, frame_id=12)
              -> trace_frame()  (dùng đúng quality user u2 đã nhận cho frame 12)
demo.py : print_trace()
tools/visualize_trace.py : save_frame_outputs()   -> 01..05, mv_overlay.png, trace.json
tools/visualize_trace.py : make_composite()      -> summary.png
```

### C. `python demo.py --web`

```
web/app.py : run_dashboard()  -> generate_sample() nếu thiếu -> _load_video(sample) -> Flask :7000

Trình duyệt:
  app.js : poll() mỗi 700 ms
    -> GET /api/state, GET /api/users
    -> vẽ lại lưới user, log LB

  Upload video   -> POST /api/upload            -> lưu, tính video_id, _load_video()
  Thêm user      -> POST /api/users             -> SessionManager.create_user()
  Kéo băng thông -> POST /api/users/<id>/bandwidth
                    (StreamingClient.bandwidth_kbps là @property, đọc lại
                     giá trị mới ở mỗi lần tải segment)
  Chạy           -> POST /api/run -> thread nền: ensure_encoded() -> start_streaming()
                    -> run_all_users() (song song cho mọi user đang có)
  Chọn user      -> GET /api/users/<id>/frames
  Chọn frame     -> GET /api/users/<id>/frames/<frame_id>
                    -> trace_user_frame() + save_frame_outputs() theo yêu cầu
                    -> ảnh trace qua GET /outputs/<path>
```

## 12. Kiểm thử

```bash
pytest
```

| Test | Nội dung |
|---|---|
| `test_I_frame`, `test_P_frame`, `test_B_frame` | phân loại + dự đoán cho từng loại frame |
| `test_encode_order` | thứ tự giải mã đúng `0,1,3,5,7,2,4,6` |
| `test_encode_decode` | round-trip encoder/decoder, DPB vòng kín nhất quán |
| `test_quant_error_bound` | $\lvert R - R' \rvert \le QP/2$ trên mọi pixel |
| `test_bitstream_integrity` | bố cục byte parse đúng bằng độ dài segment |
| `test_mv_index` | MV trong chỉ mục khớp MV đọc từ bitstream |
| `test_load_balancer` | round-robin xen kẽ + log request |
| `test_sessions` | nhiều user độc lập; đổi băng thông một user không ảnh hưởng user khác |
| `test_user_mv_lookup` | hai user cùng mức chất lượng nhận cùng MV; khác mức thì khác |
| `test_upload` | upload video, `video_id` ổn định, cache encode được dùng lại |
| `test_streaming` | node trả lời, header đúng |
| `test_sr` | resize bilinear, shape/dtype của sharpen |
| `test_end_to_end` | toàn bộ pipeline với 2 user trên clip tổng hợp nhỏ |

## 13. Vì sao các con số trông như vậy

- **I-frame nén kém nhất** (~10–25%): không có frame nào để dựa vào, chỉ có dự đoán DC nhân quả trong chính nó.
- **P-frame nén tốt** (~80%+): phần lớn khung hình khớp gần như hoàn toàn với tham chiếu sau bù chuyển động.
- **B-frame nén tốt nhất** (~90%+): trung bình hai tham chiếu làm triệt nhiễu, giống bất kỳ phép trung bình hai mẫu nào (phương sai nhiễu giảm còn khoảng một nửa).
- **User băng thông thấp có PSNR cao hơn một chút ở độ phân giải của họ** nhưng ảnh SR mờ hơn: PSNR ở đây đo ở độ phân giải nhận được, không phải chất lượng cảm nhận sau khi phóng to.
- **Ảnh phần dư có một vạch sáng dọc tại đường nối của gradient trượt**: không phải lỗi, mà là mô hình chuyển động chỉ-tịnh-tiến thừa nhận thứ nó không dự đoán được (một bước nhảy cứng).

## 14. Demo này *không* phải là gì

Không phải H.265. Không phải codec cạnh tranh. Không phải CDN thật. Không phải mô hình SR được huấn luyện. Không encode riêng cho từng user. Mọi con số in ra đều đến từ code trong repo này mà bạn có thể mở và đọc trong vài phút; khả năng truy vết đó mới là mục tiêu thật của bài tập, không phải tỷ lệ nén hay kỷ lục PSNR.#
