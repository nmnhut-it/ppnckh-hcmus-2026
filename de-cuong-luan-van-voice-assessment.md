# ĐẠI HỌC QUỐC GIA TP. HCM - TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN

**CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM**
*Độc lập - Tự do - Hạnh phúc*

---

# ĐỀ CƯƠNG ĐỀ TÀI LUẬN VĂN THẠC SĨ

## 1. Tên đề tài

**XÂY DỰNG HỆ THỐNG TỰ ĐỘNG HÓA ĐÁNH GIÁ BIẾN ĐỔI GIỌNG NÓI TIỀN VÀ HẬU PHẪU DỰA TRÊN PHÂN TÍCH ĐẶC TRƯNG ÂM HỌC ĐA CHIỀU (F0, FORMANTS VÀ PERTURBATION)**

*(Building an Automated System for Pre- and Post-Surgical Voice Change Assessment Based on Multidimensional Acoustic Feature Analysis: F0, Formants, and Perturbation)*

## 2. Ngành và mã ngành đào tạo

- **Ngành:** Khoa học máy tính
- **Mã ngành:** 60.48.01.01

## 3. Thông tin học viên và giảng viên

| Vai trò | Thông tin |
|---------|-----------|
| **Học viên** | *(Họ tên)* - Khóa: *(...)* - Đợt: *(...)* |
| **MSHV** | *(...)* |
| **Email** | *(...)* |
| **Điện thoại** | *(...)* |
| **GVHD** | *(Họ tên, học hàm, học vị)* |
| **Cơ quan** | *(...)* |
| **Email GVHD** | *(...)* |
| **ĐT GVHD** | *(...)* |

## 4. Tổng quan tình hình nghiên cứu và giới thiệu bài toán

### 4.1. Bối cảnh thực tiễn

Các bệnh lý thanh quản như polyp dây thanh, u nang dây thanh, liệt dây thanh và ung thư thanh quản ảnh hưởng trực tiếp đến chất lượng giọng nói của bệnh nhân. Phẫu thuật thanh quản (laryngeal surgery) và phẫu thuật tuyến giáp (thyroidectomy) là những can thiệp phổ biến; tuy nhiên, lên đến 80% bệnh nhân gặp phải thay đổi giọng nói tạm thời hoặc vĩnh viễn sau phẫu thuật [14]. Việc đánh giá khách quan mức độ biến đổi giọng nói trước và sau mổ là nhu cầu cấp thiết để:

- Theo dõi tiến trình phục hồi chức năng phát âm của bệnh nhân.
- Đánh giá hiệu quả can thiệp phẫu thuật một cách định lượng.
- Hỗ trợ bác sĩ lâm sàng đưa ra quyết định điều trị và liệu pháp âm ngữ phù hợp.

Hiện nay, phương pháp đánh giá giọng nói lâm sàng chủ yếu dựa trên thang đo cảm quan GRBAS (Grade-Roughness-Breathiness-Asthenia-Strain) hoặc CAPE-V, vốn mang tính chủ quan và phụ thuộc nhiều vào kinh nghiệm của chuyên gia [12]. Trong khi đó, các tham số âm học khách quan như tần số cơ bản (F0), các đỉnh cộng hưởng (Formants F1, F2, F3), các chỉ số nhiễu loạn (Jitter, Shimmer, HNR) và Cepstral Peak Prominence (CPP) đã được chứng minh là các chỉ dấu đáng tin cậy cho rối loạn giọng nói [1, 2, 3].

### 4.2. Tình hình nghiên cứu trên thế giới

**Phân tích đặc trưng âm học cho rối loạn giọng nói:**
Heman-Ackah và cộng sự (2003) đã chứng minh rằng CPP là chỉ số dự đoán khàn giọng (dysphonia) đáng tin cậy hơn so với các thước đo truyền thống về Jitter, Shimmer và NHR [1]. Fraile và Godino-Llorente (2014) phân tích toàn diện thuật toán tính CPP, cho thấy ngưỡng CPPS = 4.0 đạt độ nhạy 92.4% và độ đặc hiệu 79% trong phát hiện khàn giọng [3]. Maryn và Weenink (2015) đã xác nhận tính hợp lệ của việc tính CPPS và chỉ số AVQI (Acoustic Voice Quality Index) trong phần mềm Praat, mở ra khả năng tiếp cận miễn phí cho bác sĩ và nhà nghiên cứu [2].

**Không gian nguyên âm (Vowel Space Area - VSA) và chỉ số cấu âm nguyên âm (VAI):**
Sapir và cộng sự (2010) đề xuất Formant Centralization Ratio (FCR) cùng với VSA và VAI như các thước đo âm học cho rối loạn vận ngôn (dysarthria), tính toán từ F1 và F2 của các nguyên âm góc /a/, /i/, /u/ [4]. Whitfield và Goberman (2020) so sánh ba chỉ số VSA, FCR, VAI và chỉ ra rằng VAI là chỉ số ổn định nhất qua các khoảng thời gian ghi âm khác nhau, đồng thời nhạy nhất với các biến đổi phát âm [5]. Rusz và cộng sự (2021) phát triển pipeline tự động đánh giá cấu âm nguyên âm sử dụng bộ nhận dạng âm vị phổ quát ngôn ngữ, đạt tương quan cao với chú thích thủ công [6].

**Học máy và học sâu trong phát hiện bệnh lý giọng nói:**
Mohammed và cộng sự (2020) áp dụng CNN trên phổ tần (spectrogram) từ cơ sở dữ liệu SVD, đạt độ chính xác 95.41% [7]. Ksibi và cộng sự (2023) đề xuất bộ phân lớp hai tầng kết hợp CNN-RNN, phân loại giới tính trước rồi phân loại bệnh lý, cải thiện đáng kể so với các phương pháp truyền thống [8]. Al-Nasheri và cộng sự (2017) đánh giá các đặc trưng MDVP trên ba cơ sở dữ liệu SVD, MEEI và AVPD, đạt 99.68% trên SVD nhưng chỉ 72.53% trên AVPD, cho thấy sự ảnh hưởng lớn của dữ liệu lên kết quả [9]. Tổng quan hệ thống của Barlow và cộng sự (2024) trên 82 nghiên cứu cho thấy CNN hiệu quả nhất trên hình ảnh nội soi thanh quản, trong khi MLP hiệu quả nhất trên đầu vào âm học [11].

**Đánh giá giọng nói trước và sau phẫu thuật:**
Sinagra và cộng sự (2011) phân tích giọng nói trước/sau cắt tuyến giáp toàn phần, chỉ ra rằng khi không có tổn thương thần kinh thanh quản quặt ngược, các thay đổi là tối thiểu và tạm thời [12]. Stager và cộng sự (2025) sử dụng AVQI qua Praat để đánh giá tiền cứu, nhận thấy CPP của nguyên âm giảm có ý nghĩa thống kê sau phẫu thuật tuyến giáp ở nữ giới [13]. Lechien và cộng sự (2024) nhấn mạnh phương pháp đánh giá đa chiều kết hợp phân tích cepstral, đánh giá cảm quan và thang đo tự đánh giá của bệnh nhân [15].

### 4.3. Khoảng trống nghiên cứu

Mặc dù đã có nhiều tiến bộ, các hệ thống hiện tại vẫn tồn tại một số hạn chế:

- **Thiếu tính tự động hóa toàn diện:** Hầu hết quy trình đánh giá vẫn yêu cầu can thiệp thủ công đáng kể từ chuyên gia.
- **Thiếu phân tích dọc (longitudinal):** Đa số nghiên cứu tập trung phát hiện bệnh lý tại một thời điểm; rất ít công trình phân tích biến đổi theo cặp (paired) tiền-hậu phẫu.
- **Thiếu tích hợp đa chiều:** Các phương pháp thường chỉ sử dụng một nhóm đặc trưng (perturbation HOẶC formant HOẶC cepstral), chưa khai thác tổ hợp đầy đủ.
- **Khoảng trống về ngôn ngữ:** Tiếng Việt là ngôn ngữ thanh điệu (6 thanh), biến thiên F0 khác biệt cơ bản so với các ngôn ngữ châu Âu trong các cơ sở dữ liệu hiện có (SVD, AVFAD). Chưa có cơ sở dữ liệu bệnh lý giọng nói tiếng Việt.

Do đó, việc xây dựng một hệ thống tự động hóa đánh giá biến đổi giọng nói tiền và hậu phẫu, tích hợp phân tích đặc trưng âm học đa chiều, là **nội dung và mục đích của đề tài này**.

## 5. Tính khoa học và tính cấp thiết của đề tài

### Tính khoa học

- Đề xuất pipeline xử lý tín hiệu âm thanh tự động trích xuất đồng thời ba nhóm đặc trưng: **nguồn âm** (F0, F0SD), **bộ lọc** (F1, F2, F3), và **nhiễu loạn** (Jitter, Shimmer, HNR, CPP).
- Áp dụng mô hình hóa không gian nguyên âm (Convex Hull trên mặt phẳng F1-F2) để tính **Vowel Space Area (VSA)** và **Vowel Articulation Index (VAI)** -- các chỉ số phản ánh chức năng cấu âm tổng thể.
- Thiết kế phương pháp phân tích biến đổi dọc (longitudinal) thông qua vector sai lệch **V = V_T1 - V_T0** để định lượng mức độ thay đổi chức năng phát âm trước và sau phẫu thuật.
- Xây dựng mô hình phân lớp kết hợp học máy (SVM, Random Forest) và/hoặc học sâu (CNN trên Mel-spectrogram, MLP trên vector đặc trưng) để phân loại mức độ bệnh lý giọng nói với độ chính xác mục tiêu > 90%.

### Tính cấp thiết

- Tại Việt Nam, nhu cầu đánh giá giọng nói lâm sàng ngày càng tăng trong khi số lượng chuyên gia âm ngữ trị liệu còn rất hạn chế (tính đến 2015, chỉ có khoảng 33 nhà trị liệu ngôn ngữ có chứng chỉ trên cả nước).
- Hệ thống tự động hóa sẽ hỗ trợ đắc lực cho các bác sĩ tai mũi họng và phẫu thuật viên trong việc theo dõi tiến trình phục hồi sau mổ, giảm sự phụ thuộc vào đánh giá chủ quan.
- Camera giám sát và thu âm y tế ngày càng phổ biến tại các bệnh viện lớn ở Việt Nam, cung cấp nguồn dữ liệu âm thanh có thể khai thác cho mục đích chẩn đoán hỗ trợ.

### Tính mới

- **Phân tích đa chiều đồng thời:** Kết hợp cả ba nhóm đặc trưng (source, filter, perturbation) cùng VSA/VAI trong một pipeline thống nhất -- khác với các nghiên cứu hiện tại thường chỉ dùng một nhóm.
- **Phân tích dọc (paired pre/post):** Thiết kế đặc biệt cho so sánh tiền-hậu phẫu theo cặp, tận dụng đặc tính intra-subject để giảm biến thiên inter-subject.
- **Hướng đến ứng dụng cho ngôn ngữ thanh điệu:** Mở đường cho việc áp dụng hệ thống trên dữ liệu tiếng Việt trong tương lai.

## 6. Mục tiêu, đối tượng và phạm vi nghiên cứu

### Mục tiêu

1. Xây dựng pipeline tự động trích xuất đặc trưng âm học đa chiều (F0, F0SD, F1, F2, F3, Jitter, Shimmer, HNR, CPP) từ tín hiệu giọng nói nguyên âm kéo dài.
2. Phát triển module tính toán VSA và VAI dựa trên thuật toán Convex Hull trên mặt phẳng F1-F2.
3. Thiết kế phương pháp phân tích biến đổi dọc (longitudinal) định lượng sự thay đổi giọng nói giữa hai thời điểm T0 và T1.
4. Xây dựng mô hình phân lớp (Healthy vs. Pathological) đạt độ chính xác > 90% trên các cơ sở dữ liệu chuẩn.
5. Phát triển hệ thống chẩn đoán hỗ trợ (CAD) với giao diện trực quan hóa (VSA map, Radar chart) phục vụ bác sĩ lâm sàng.

### Đối tượng nghiên cứu

- **Tín hiệu âm thanh:** Bản ghi âm nguyên âm kéo dài /a/, /e/, /i/, /o/, /u/ của bệnh nhân trước và sau phẫu thuật thanh quản/tuyến giáp.
- **Đặc trưng âm học:** F0, F0SD, F1, F2, F3, Jitter (local), Shimmer (local), HNR, CPP/CPPS, VSA, VAI, FCR.
- **Phương pháp học máy:** SVM, Random Forest, MLP, và CNN trên biểu diễn phổ (Mel-spectrogram).

### Phạm vi nghiên cứu

- Sử dụng các cơ sở dữ liệu công khai quốc tế (chi tiết tại Mục 7.1) để huấn luyện và đánh giá mô hình, không yêu cầu thu thập dữ liệu lâm sàng mới (tránh vấn đề đạo đức nghiên cứu).
- Giới hạn phân tích trên nguyên âm kéo dài (sustained vowels), chưa xử lý giọng nói liên tục (connected speech) hoặc giọng hát.
- Hệ thống CAD mang tính hỗ trợ chẩn đoán, không thay thế quyết định lâm sàng của bác sĩ.

## 7. Nội dung, phương pháp nghiên cứu

### 7.1. Cơ sở dữ liệu nghiên cứu

| Cơ sở dữ liệu | Quy mô | Nguyên âm | Đặc điểm nổi bật | Nguồn |
|----------------|--------|-----------|-------------------|-------|
| **Saarbrücken Voice Database (SVD)** | 2.000+ người (1.002 bệnh lý, 851 khỏe mạnh, 71 loại rối loạn) | /a/, /i/, /u/ ở nhiều cao độ | Toàn diện nhất, tần số lấy mẫu 50 kHz | [stimmdb.coli.uni-saarland.de](https://stimmdb.coli.uni-saarland.de/) |
| **AVFAD** | 709 người (346 bệnh lý, 363 khỏe mạnh) | /a/, /e/, /o/ | Cung cấp sẵn 19 tham số âm học từ Praat | [ResearchGate](https://www.researchgate.net/publication/319876992) |
| **VOICED (PhysioNet)** | 208 mẫu (150 bệnh lý, 58 khỏe mạnh) | /a/ kéo dài 5 giây | Thu âm bằng smartphone (Samsung Galaxy S4) | [physionet.org](https://physionet.org/content/voiced/1.0.0/) |
| **FEMH Voice Dataset** | 2.000 người (bao gồm ung thư thanh quản, liệt dây thanh) | /a/ | Dành cho các bài toán thử thách phân loại | [ResearchGate](https://www.researchgate.net/publication/330629543) |

### 7.2. Nội dung nghiên cứu

#### A. Tiền xử lý tín hiệu (Signal Preprocessing)

- Loại bỏ nhiễu nền, chuẩn hóa biên độ (amplitude normalization).
- Phân đoạn (segmentation) lấy phần ổn định (steady-state) của nguyên âm kéo dài, loại bỏ onset và offset.
- Kiểm tra chất lượng tín hiệu (SNR > 30 dB) để đảm bảo tính tin cậy của phân tích.

#### B. Trích xuất đặc trưng âm học đa chiều (Multidimensional Feature Extraction)

Sử dụng thư viện **Parselmouth** (giao diện Python cho Praat) [20] để trích xuất:

| Nhóm đặc trưng | Tham số | Ý nghĩa lâm sàng |
|-----------------|---------|-------------------|
| **Nguồn âm (Source)** | F0 (Hz), F0SD (Hz) | Tần số dao động dây thanh; biến thiên cao gợi ý mất kiểm soát |
| **Bộ lọc (Filter)** | F1, F2, F3 (Hz) | Hình dạng đường thanh đạo; phản ánh chức năng cấu âm |
| **Nhiễu loạn (Perturbation)** | Jitter (%), Shimmer (%), HNR (dB), CPP (dB) | Tính bất ổn chu kỳ/biên độ; tỷ lệ hài/nhiễu; đỉnh cepstral |

#### C. Mô hình hóa không gian nguyên âm (Vowel Space Modeling)

- Chiếu các nguyên âm /a, e, i, o, u/ lên mặt phẳng F1-F2 (F1 = trục ngang, F2 = trục dọc).
- Tính diện tích đa giác lồi (Convex Hull Area) bao quanh các điểm nguyên âm => **VSA** (đơn vị: Hz²).
- Tính **VAI** = (F2/i/ + F1/a/) / (F1/i/ + F1/u/ + F2/u/ + F2/a/) theo công thức Sapir và cộng sự [4].
- Tính **FCR** = nghịch đảo của VAI để đánh giá mức độ tập trung hóa formant.

#### D. Phân tích biến đổi dọc (Longitudinal Analysis)

- Với mỗi bệnh nhân, xây dựng vector đặc trưng tại T0 và T1:
  - **V_T0** = [F0, F0SD, F1_a, F2_a, ..., Jitter, Shimmer, HNR, CPP, VSA, VAI]
  - **V_T1** = vector tương ứng sau phẫu thuật
- Tính **vector sai lệch V = V_T1 - V_T0** để định lượng mức độ thay đổi.
- Áp dụng kiểm định thống kê (paired t-test, Wilcoxon signed-rank test) để xác định ý nghĩa thống kê của biến đổi.
- Tính Effect Size (Cohen's d) để đánh giá ý nghĩa lâm sàng.

#### E. Mô hình phân lớp (Classification Model)

Xây dựng và so sánh các mô hình:

| Mô hình | Đầu vào | Mô tả |
|---------|---------|-------|
| **SVM (RBF kernel)** | Vector đặc trưng âm học | Baseline mạnh, được sử dụng phổ biến nhất trong lĩnh vực (35.2% các nghiên cứu) [22] |
| **Random Forest** | Vector đặc trưng âm học | Khả năng diễn giải cao, xác định feature importance |
| **MLP (Multi-Layer Perceptron)** | Vector đặc trưng âm học | Hiệu quả nhất cho đầu vào âm học theo [11] |
| **CNN** | Mel-spectrogram | Khai thác biểu diễn thời gian-tần số, hiệu quả cho hình ảnh phổ [7] |

- Phân chia dữ liệu: 70% train, 15% validation, 15% test (stratified split).
- Đánh giá: Accuracy, Precision, Recall, F1-Score, AUC-ROC.
- Cross-validation: Stratified 5-fold hoặc Leave-One-Speaker-Out (LOSO).
- Mục tiêu: Accuracy > 90%, AUC > 0.95 trên phân loại Healthy vs. Pathological.

#### F. Hệ thống chẩn đoán hỗ trợ (CAD System)

Xây dựng ứng dụng với các tính năng:

- **Đầu vào:** Upload file âm thanh (WAV, MP3) của 5 nguyên âm /a, e, i, o, u/ tại T0 và T1.
- **Trực quan hóa:**
  - VSA map: Biểu đồ không gian nguyên âm F1-F2 so sánh T0 vs. T1.
  - Radar chart: Biểu đồ radar đa chiều hiển thị tất cả tham số âm học.
  - Bảng so sánh chi tiết từng tham số với chỉ thị tăng/giảm và mức ý nghĩa.
- **Đầu ra phân lớp:** Kết quả Healthy vs. Pathological kèm xác suất tin cậy.
- **Báo cáo:** Tạo báo cáo PDF tự động phục vụ hồ sơ y tế.

### 7.3. Công cụ và công nghệ triển khai

| Công cụ | Vai trò |
|---------|---------|
| **Python 3.10+** | Ngôn ngữ lập trình chính |
| **Parselmouth** | Giao diện Python cho Praat -- trích xuất F0, F1, F2, F3 chính xác theo tiêu chuẩn y khoa [20] |
| **Librosa** | Tiền xử lý và trích xuất đặc trưng phổ (Mel-spectrogram, MFCC) |
| **SciPy** | Thuật toán Convex Hull cho VSA, kiểm định thống kê |
| **Scikit-learn** | Xây dựng mô hình SVM, Random Forest, MLP; đánh giá mô hình |
| **PyTorch** | Xây dựng mô hình CNN trên Mel-spectrogram |
| **Streamlit / Flask** | Phát triển giao diện web cho hệ thống CAD |
| **Matplotlib / Plotly** | Trực quan hóa (VSA map, Radar chart) |

### 7.4. Phương pháp nghiên cứu

- **Phương pháp khảo sát, phân tích đánh giá:** Tìm hiểu các nghiên cứu liên quan đến phân tích âm học giọng nói bệnh lý, đánh giá giọng nói tiền-hậu phẫu, và các phương pháp học máy trong lĩnh vực. Phân tích ưu nhược điểm của từng phương pháp.
- **Phương pháp thực nghiệm:** Triển khai pipeline trên các cơ sở dữ liệu chuẩn quốc tế (SVD, AVFAD, VOICED, FEMH), so sánh kết quả với các công trình đã công bố.
- **Phương pháp tổng hợp và mô hình hóa:** Trực quan hóa kết quả, xây dựng mô hình logic cho hệ thống CAD.
- **Phương pháp thử và sai:** Thử nghiệm nhiều tổ hợp đặc trưng, nhiều kiến trúc mô hình, tối ưu siêu tham số (hyperparameter tuning) để chọn cấu hình tốt nhất.

### 7.5. Dự kiến kết quả đạt được

1. **Bài báo khoa học:** Xuất bản ít nhất một bài báo tại hội nghị/tạp chí trong nước hoặc quốc tế trình bày về phương pháp đánh giá biến đổi giọng nói đa chiều.
2. **Pipeline tự động:** Hệ thống trích xuất đặc trưng và phân lớp đạt Accuracy > 90% trên ít nhất hai cơ sở dữ liệu chuẩn.
3. **Phân tích dọc:** Phương pháp định lượng biến đổi tiền-hậu phẫu với kiểm định thống kê.
4. **Ứng dụng CAD:** Prototype hệ thống với giao diện web, hỗ trợ upload âm thanh, trích xuất đặc trưng tự động, trực quan hóa, và phân lớp.

### 7.6. Hướng phát triển

- Thu thập và xây dựng cơ sở dữ liệu bệnh lý giọng nói tiếng Việt (các nguyên âm tiếng Việt với 6 thanh điệu).
- Mở rộng phân tích sang giọng nói liên tục (connected speech) và giọng hát.
- Tích hợp phân tích hình ảnh nội soi thanh quản (laryngoscopy) kết hợp với phân tích âm học để tạo hệ thống multimodal.
- Phát triển ứng dụng di động thu âm và đánh giá giọng nói tại chỗ (point-of-care), phù hợp với bối cảnh y tế Việt Nam.
- Ứng dụng transfer learning từ mô hình huấn luyện trên dữ liệu quốc tế sang dữ liệu tiếng Việt.

## 8. Kế hoạch bố trí thời gian nghiên cứu

| Thời gian | Giai đoạn | Nội dung |
|-----------|-----------|----------|
| 01 tháng | **Giai đoạn 1** | Khảo sát tổng quan lý thuyết: các phương pháp phân tích âm học giọng nói, các mô hình học máy cho phát hiện bệnh lý giọng nói, các phương pháp đánh giá giọng nói tiền-hậu phẫu. Tải về và khám phá các cơ sở dữ liệu SVD, AVFAD, VOICED, FEMH. |
| 02 tháng | **Giai đoạn 2** | Xây dựng pipeline trích xuất đặc trưng âm học (F0, Formants, Perturbation) sử dụng Parselmouth/Librosa. Triển khai module tính VSA, VAI. Cài đặt và huấn luyện các mô hình phân lớp (SVM, RF, MLP, CNN). |
| 02 tháng | **Giai đoạn 3** | Thực nghiệm phân tích biến đổi dọc (longitudinal). Tối ưu hóa mô hình, so sánh hiệu năng. Phát triển hệ thống CAD (giao diện web, trực quan hóa). Cải tiến và nâng cao hiệu suất. |
| 01 tháng | **Giai đoạn 4** | Viết luận văn thạc sĩ. |
| 0,5 tháng | **Giai đoạn 5** | Báo cáo và bảo vệ luận văn thạc sĩ. |
| 03 tháng | **Song song** | Viết 1 bài báo khoa học cho hội nghị/tạp chí chuyên ngành (phải hoàn thành trước giai đoạn 5). |

## 9. Tài liệu tham khảo

[1] Y. D. Heman-Ackah, D. D. Michael, M. M. Baroody, et al., "Cepstral Peak Prominence: A More Reliable Measure of Dysphonia," *Annals of Otology, Rhinology & Laryngology*, vol. 112, no. 4, pp. 324-333, 2003.

[2] Y. Maryn and D. Weenink, "Objective Dysphonia Measures in the Program Praat: Smoothed Cepstral Peak Prominence and Acoustic Voice Quality Index," *Journal of Voice*, vol. 29, pp. 35-43, 2015.

[3] R. Fraile and J. I. Godino-Llorente, "Cepstral Peak Prominence: A Comprehensive Analysis," *Biomedical Signal Processing and Control*, vol. 14, pp. 42-54, 2014.

[4] S. Sapir, L. O. Ramig, J. L. Spielman, and C. Fox, "Formant Centralization Ratio: A Proposal for a New Acoustic Measure of Dysarthric Speech," *Journal of Speech, Language, and Hearing Research*, vol. 53, pp. 114-125, 2010.

[5] J. A. Whitfield and A. M. Goberman, "Stability, Reliability, and Sensitivity of Acoustic Measures of Vowel Space," *Journal of the Acoustical Society of America*, vol. 148, no. 3, 2020.

[6] J. Rusz, T. Tykalova, M. Novotny, et al., "Defining Speech Subtypes in De Novo Parkinson's Disease: Response to Long-Term Levodopa Therapy," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 1-13, 2021.

[7] M. A. Mohammed, K. H. Abdulkareem, S. A. Mostafa, et al., "Voice Pathology Detection and Classification Using Convolutional Neural Network Model," *Applied Sciences*, vol. 10, no. 11, 3723, 2020.

[8] A. Ksibi, N. A. Hakami, N. Alturki, et al., "Voice Pathology Detection Using a Two-Level Classifier Based on Combined CNN-RNN Architecture," *Sustainability*, vol. 15, no. 4, 3204, 2023.

[9] A. Al-Nasheri, G. Muhammad, M. Alsulaiman, et al., "An Investigation of Multidimensional Voice Program Parameters in Three Different Databases for Voice Pathology Detection and Classification," *Journal of Voice*, vol. 31, no. 1, pp. 113.e9-113.e18, 2017.

[10] J. I. Godino-Llorente, P. Gomez-Vilda, and M. Blanco-Velasco, "Dimensionality Reduction of a Pathological Voice Quality Assessment System Based on Gaussian Mixture Models and Short-Term Cepstral Parameters," *IEEE Transactions on Biomedical Engineering*, vol. 53, no. 10, pp. 1943-1953, 2006.

[11] J. Barlow, Z. Sragi, L. Rivera-Rivera, et al., "The Use of Deep Learning Software in the Detection of Voice Disorders: A Systematic Review," *Otolaryngology-Head and Neck Surgery*, vol. 170, no. 6, 2024.

[12] D. G. Sinagra, G. Montesano, and F. Dispenza, "Perceptual and Acoustic Analysis of Voice in Individuals with Total Thyroidectomy: Pre-Post Surgery Comparison," *Indian Journal of Otolaryngology and Head & Neck Surgery*, 2011.

[13] S. Stager, S. Bielamowicz, and T. Guelrud, "Prospective Voice Assessment After Thyroidectomy Without Recurrent Laryngeal Nerve Injury," *Head & Neck*, 2025.

[14] S. Yun, D. Kim, and H. S. Choi, "Voice Outcomes as Results of Voice Therapy after Lobectomy and Thyroidectomy," *Journal of Voice*, 2021.

[15] J. R. Lechien, et al., "Early Assessment of Voice Problems in Post-Thyroidectomy Syndrome Using Cepstral Analysis," *Journal of Clinical Medicine*, 2024.

[16] D. Martinez, E. Lleida, A. Ortega, et al., "Voice Pathology Detection on the Saarbrucken Voice Database with Calibration and Fusion of Scores Using MultiFocal Toolkit," *Advances in Speech and Language Technologies for Iberian Languages (IberSPEECH)*, LNAI 7911, pp. 99-108, Springer, 2012.

[17] L. M. T. Jesus, I. Belo, J. Machado, and A. Hall, "The Advanced Voice Function Assessment Databases (AVFAD)," *Advances in Speech-Language Pathology*, IntechOpen, 2017.

[18] S. H. Fang, Y. Tsao, M. J. Hsiao, et al., "Detection of Pathological Voice Using Cepstrum Vectors: A Deep Learning Approach," *IEEE International Conference on Big Data*, Seattle, WA, 2019.

[19] P. Boersma and D. Weenink, *Praat: doing phonetics by computer* [Computer program], Version 6.4, 2024. http://www.praat.org/

[20] Y. Jadoul, B. Thompson, and B. de Boer, "Introducing Parselmouth: A Python Interface to Praat," *Journal of Phonetics*, vol. 71, pp. 1-15, 2018.

[21] Y. Liu, et al., "A Scoping Review of Artificial Intelligence Detection of Voice Pathology: Challenges and Opportunities," *Otolaryngology-Head and Neck Surgery*, 2024.

[22] A. Idrisoglu, A. L. Dallora, P. Anderberg, and J. S. Berglund, "Applied Machine Learning Techniques to Diagnose Voice-Affecting Conditions and Disorders: Systematic Literature Review," *Journal of Medical Internet Research*, vol. 25, e46105, 2023.

---

| **Giảng viên hướng dẫn** | **Học viên cao học** |
|---|---|
| *(Ký tên)* | *(Ký tên)* |
