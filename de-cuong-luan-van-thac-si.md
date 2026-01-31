# ĐẠI HỌC QUỐC GIA TP. HCM - TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN

**CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM**
*Độc lập - Tự do - Hạnh phúc*

---

# ĐỀ CƯƠNG ĐỀ TÀI LUẬN VĂN THẠC SĨ

## 1. Tên đề tài

**PHÁT TRIỂN MÔ HÌNH KẾT HỢP CHO BÀI TOÁN THEO VẾT ĐỐI TƯỢNG**

*(Developing a Combined Model for Object Tracking)*

## 2. Ngành và mã ngành đào tạo

- **Ngành:** Khoa học máy tính
- **Mã ngành:** 60.48.01.01

## 3. Thông tin học viên và giảng viên

| Vai trò | Thông tin |
|---------|-----------|
| **Học viên** | Lê Quang Thái - Khóa: 10 - Đợt: 1 |
| **MSHV** | CH1501015 |
| **Email** | quangthai121121@gmail.com |
| **Điện thoại** | 0939 792 871 |
| **GVHD** | PGS. TS. Lê Hoàng Thái |
| **Cơ quan** | Khoa Công nghệ Thông tin, ĐH Khoa học Tự nhiên |
| **Email GVHD** | lhthai@hcmus.edu.vn |
| **ĐT GVHD** | 0983 497 225 |

## 4. Tổng quan tình hình nghiên cứu và giới thiệu bài toán

Với sự phát triển của khoa học công nghệ như hiện nay, ta dễ dàng có được những bức ảnh chất lượng có khung hình đẹp mà dung lượng không quá lớn. Các thiết bị ghi hình có thể thấy và hiểu được môi trường xung quanh đã được xây dựng và phát triển ngày càng nhiều bởi sự tiến bộ trong vi điện tử cũng như trong các thuật toán phân tích video. Hiện tại, nhiều cơ hội được mở ra để phát triển các ứng dụng trong nhiều lĩnh vực như giám sát video, sáng tạo nội dung, liên lạc cá nhân hay tương tác giữa người và máy. Trong đó, một tính năng cơ bản và cần thiết để máy móc có thể xem, hiểu và phản ứng với môi trường xung quanh chính là khả năng phát hiện và theo vết đối tượng mà ta quan tâm. Quá trình ước lượng vị trí của một hay nhiều đối tượng nào đó theo thời gian qua camera, được gọi là **video tracking**.

> **Hình 1:** Các hệ thống giám sát thông minh bằng hình ảnh.

> **Hình 2:** Hệ thống bãi giữ xe thông minh. [Công ty TNHH giám sát 24h].

Hiện nay, các hệ thống giám sát thông minh bằng hình ảnh trên thế giới đã được phát triển và chứng minh được tính hiệu quả nhất định trên một số lĩnh vực từ theo dõi an ninh, nhận dạng đối tượng cho đến giám sát giao thông.

Một số ứng dụng ta có thể thường gặp trong cuộc sống hằng ngày như:

- **Nhận dạng đường đi** của đối tượng, qua đó có thể xác định được đối tượng hoặc danh tính của con người dựa vào dáng đi và đặc điểm khuôn mặt.
- **Tự động giám sát**, nhận diện và ghi hình các hoạt động thuộc diện khả nghi phục vụ nhu cầu an ninh ở sân bay hoặc trong các toà nhà cao tầng, khu chung cư và có thể hỗ trợ chức năng tìm kiếm và truy xuất hiệu quả hình ảnh trong quá khứ.
- **Giám sát giao thông**: hệ thống quan sát, thống kê tình trạng giao thông thời gian thực nhằm có phương pháp điều tiết giao thông hợp lý trong những giờ cao điểm.
- **Thông thương hàng hải**: hệ thống giám sát thông minh giúp xác định kế hoạch đường đi để tránh khả năng gây trở ngại.

Tuy nhiên, các hệ thống trên vẫn gặp phải một số hạn chế như hiệu quả của việc quan sát luôn phụ thuộc vào điều kiện môi trường quan sát, kiểu chuyển động của đối tượng. Do đó, một bài toán quan trọng đặt ra để hoàn thiện hệ thống giám sát thông minh là **bài toán theo vết đối tượng tự động**.

### Các phương pháp theo vết đối tượng tiêu biểu

Trong những năm gần đây, bài toán theo vết đối tượng đã có nhiều phương pháp được nghiên cứu và phát triển:

1. **Sparsity-based Collaborative Model (SCM)** [1]: Là một phần của Visual Object Tracking Repository, có tính ổn định cao khi thực hiện quá trình theo vết.

2. **Structured Output Tracking (STR)** [2]: Sử dụng thuật toán SVM (Support Vector Machine) để theo vết đối tượng.

3. **Tracking-Learning-Detection (TLD)** [3,4]: Sử dụng phương thức phát hiện-theo vết, do đó quá trình phát hiện và theo dõi được thực hiện cùng lúc với nhau. Tuy nhiên chỉ thích hợp với các đối tượng không biến mất khỏi màn hình theo dõi.

4. **Compressive Tracking (CT)** [5]: Thuật toán thực hiện nhiệm vụ theo vết ở thời gian thực và đáp ứng các điều kiện thử thách về mặt hiệu quả cũng như sự chính xác.

5. **Clustering of Static-Adaptive Correspondences for Deformable Object Tracking (CMT)** [6]: Sử dụng phương pháp phân lớp theo trọng tâm với các điểm quan trọng bên trong và bên ngoài nên có được độ chính xác cao.

> **Hình 3:** Theo vết đối tượng bằng phương pháp CMT [6].

> **Hình 4:** Theo vết đối tượng bằng phương pháp TLD [3].

Các phương pháp trên mặc dù có những thành công trong việc phát hiện và theo vết đối tượng, tuy nhiên vẫn còn những hạn chế tồn tại. Do đó việc kết hợp các phương pháp theo vết đã có để đề xuất một mô hình theo vết đối tượng quan tâm cho một mục đích nào đó là **nội dung và mục đích của đề tài này**.

## 5. Tính khoa học và tính cấp thiết của đề tài

### Tính khoa học

Tính khoa học của luận văn thể hiện qua việc đề xuất một phương pháp theo vết đối tượng mới bằng cách kết hợp các phương pháp theo vết đối tượng đã có. Bước đầu của quá trình kết hợp các phương pháp theo vết sẵn có đã được thể hiện trong bài báo *"A Fusion TLD and CMT Model for Motion Object Tracking"* được trình bày ở hội nghị **ICISCA 2016**, July 2016 [7].

Thực hiện nghiên cứu, thử nghiệm, đánh giá và cải tiến phương pháp kết hợp đề xuất để phù hợp với nhu cầu theo vết đối tượng của các hệ thống theo dõi ở Việt Nam hiện nay.

### Tính cấp thiết

Hiện nay, hệ thống camera giám sát rất phổ biến, chúng thu thập dữ liệu hình ảnh ở mọi nơi, mọi lúc. Với lượng dữ liệu to lớn như vậy, nhu cầu được đặt ra cần phải khai thác, xử lý và kết xuất được những thông tin hữu ích từ những dữ liệu đã có nhằm phục vụ cho các mục đích an ninh, phân tích, thống kê, phục vụ cho các hoạt động khoa học khác.

Tích hợp phương pháp kết hợp vào những hệ thống có nhu cầu quản lý tiềm năng hiện nay như: camera an ninh tại các toà nhà cao tầng, khu chung cư, các bãi giữ xe, văn phòng với giá trị tài sản rất lớn nhằm nâng cao hiệu quả giám sát cho hệ thống an ninh.

### Tính mới

Hầu hết các giải thuật trên thế giới chỉ áp dụng được trên các camera có độ phân giải cao hoặc 3D với bộ xử lý mạnh mẽ. Việc tìm giải thuật theo vết cho các camera có độ phân giải thấp vẫn còn hạn chế.

## 6. Mục tiêu, đối tượng và phạm vi nghiên cứu

### Mục tiêu

Mục tiêu của đề tài là phát hiện và theo vết được đối tượng theo thời gian thực trong một đoạn video hay qua camera. Dựa trên sự kết hợp hai phương pháp **TLD** và **CMT**, mô hình kết hợp sẽ được xây dựng và giải quyết bài toán theo vết đối tượng đã được đặt ra.

### Đối tượng nghiên cứu

- Nghiên cứu về các định dạng phim ảnh, các chuẩn loại phim cũng như các khái quát về xử lý video.
- Tìm hiểu các nghiên cứu liên quan về việc giám sát chuyển động của đối tượng trong video.
- Khảo sát, phân tích ưu nhược điểm của các phương pháp phát hiện và theo vết đối tượng.
- Kết hợp các mô hình tính toán trong các phương pháp sẵn có để xây dựng một tính toán thích hợp cho bài toán theo vết đối tượng chuyển động.

### Phạm vi nghiên cứu

Đề tài được thực hiện chủ yếu trong việc theo vết các đối tượng chuyển động thông qua camera. Giới hạn mô hình tính toán áp dụng cho **một camera theo vết một đối tượng tại một thời điểm**. Tập dữ liệu để thực hiện có thể là các video sẵn có hoặc tự thu thập được qua các thiết bị ngoài như camera ghi hình tại các toà nhà cao tầng.

## 7. Nội dung, phương pháp nghiên cứu

### 7.1. Nội dung nghiên cứu

#### Về lý thuyết

Một trong những thách thức khi thực hiện theo vết đối tượng là **sự hỗn loạn hình ảnh (clutter)** - sự tương đồng giữa đối tượng đó với các đối tượng khác trong khung cảnh, điều này sẽ gây khó khăn cho máy tính quan sát.

Ngoài ra việc theo vết đối tượng còn gặp một số khó khăn khác như sự xuất hiện của đối tượng đó thay đổi theo từng khung hình của đoạn video hoặc độ chính xác của quá trình theo vết còn phụ thuộc vào điều kiện môi trường bên ngoài như ánh sáng, góc độ khung hình.

Để thực hiện được việc kết hợp hai phương pháp CMT và TLD, cần phải:

- Nghiên cứu ý tưởng, mô hình của giải thuật
- Phân tích những điểm mạnh điểm yếu của cả hai giải thuật CMT và TLD
- Phát triển mô hình kết hợp tận dụng ưu thế và hạn chế khuyết điểm của cả hai phương pháp
- Thử nghiệm, đánh giá mô hình kết hợp dựa trên bộ dữ liệu Vojir và bộ dữ liệu thực tế

#### Về ứng dụng thực tế

Xây dựng một ứng dụng hỗ trợ hoạt động giám sát an ninh với các tính năng:

- Lấy nguồn dữ liệu đầu vào trực tiếp từ các camera an ninh, video online hoặc video offline đã được lưu lại
- Hỗ trợ một số định dạng video cơ bản như MOV, AVI, MPEG, MP4 và WMV
- Phát hiện đối tượng chuyển động từ tập video đầu vào
- Xuất ra danh sách khoảng thời gian đối tượng chuyển động
- Tự động chuyển đến đoạn thời gian mà đối tượng chuyển động trong video, hoặc xuất ra thông báo nếu không có đối tượng chuyển động xuất hiện trong video

### 7.2. Phương pháp nghiên cứu

- **Phương pháp khảo sát, phân tích đánh giá**: Tìm hiểu các nghiên cứu liên quan đến việc phát hiện và theo vết đối tượng, nắm bắt ý tưởng chung, phân tích các ưu và khuyết điểm của từng phương pháp.
- **Phương pháp tổng hợp và mô hình hoá**: Tổng hợp, trực quan hoá các đề xuất trong nghiên cứu thành các mô hình logic rõ ràng.
- **Phương pháp thử và sai**: Thử nghiệm mô hình đề xuất trên bộ dữ liệu video Vojir và các bộ dữ liệu thực tế để kiểm chứng và lựa chọn mô hình có tính khả thi.

### 7.3. Dự kiến kết quả đạt được

- Xuất bản một bài báo khoa học cho hội nghị/tạp chí trong nước hoặc quốc tế trình bày về mô hình kết hợp phương pháp TLD_CMT
- Tính khả thi của mô hình kết hợp được thể hiện qua tốc độ và độ chính xác chấp nhận được để theo vết đối tượng thông qua camera trong điều kiện tiêu chuẩn của Việt Nam
- **Giá trị thực tiễn**: tạo một ứng dụng theo vết đối tượng cho các camera ghi hình đặt tại các toà nhà cao tầng

### 7.4. Hướng phát triển

- Nâng cao độ chính xác và khắc phục một số hạn chế của giải thuật
- Phát triển mô hình ứng dụng cho hệ thống phức tạp gồm nhiều camera, đồng thời có khả năng theo vết nhiều đối tượng cùng lúc
- Hoàn thiện ứng dụng để có thể áp dụng vào môi trường thực tế ở Việt Nam

## 8. Kế hoạch bố trí thời gian nghiên cứu

| Thời gian | Giai đoạn | Nội dung |
|-----------|-----------|----------|
| 0,5 tháng | Giai đoạn 1 | Khảo sát tổng quan về bài toán theo vết đối tượng và các phương pháp hiện tại. Tìm hiểu phương pháp TLD |
| 02 tháng | Giai đoạn 2 | Xây dựng mô hình, cài đặt môi trường, thực nghiệm |
| 02 tháng | Giai đoạn 3 | Đưa ra các cải tiến nhằm nâng cao hiệu suất cho giải thuật |
| 01 tháng | Giai đoạn 4 | Viết khóa luận thạc sĩ |
| 0,5 tháng | Giai đoạn 5 | Báo cáo khóa luận thạc sĩ |
| 03 tháng | Song song | Viết 1 bài báo khoa học cho hội nghị/tạp chí chuyên ngành (phải thực hiện trước giai đoạn 5) |

## 9. Tài liệu tham khảo

1. Wei Zhong, *Robust object tracking via sparsity-based collaborative model*, Proceedings / CVPR, IEEE Computer Society Conference on Computer Vision and Pattern Recognition, June 2012.
2. Sam Hare, Amir Saffari, and Philip H. S. Torr, *Struck: Structured Output Tracking with Kernels*, International Conference on Computer Vision (ICCV), 2011.
3. Georg Nebehay, *Robust Object Tracking Based on Tracking-Learning-Detection*, Thesis.
4. Zdenek Kalal, Krystian Mikolajczyk, Jiri Matas, *Tracking-Learning-Detection*, Journal IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 34 Issue 7, July 2012, pp. 1409-1422.
5. Kaihua Zhang, Lei Zhang, Ming-Hsuan Yang, *Real-Time Compressive Tracking*, Computer Vision ECCV 2012, Volume 7574 of the series Lecture Notes in Computer Science, pp. 864-877.
6. Georg Nebehay, Roman Pflugfelder, *Clustering of Static-Adaptive Correspondences for Deformable Object Tracking*, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 2784-2791.
7. An Tien Vo, Thai Quang Le, Hai Son Tran, Thai Hoang Le, *A Fusion TLD and CMT Model for Motion Object Tracking*, International Conference on Information, System and Convergence Applications, Vol 3, No.1, July 2016, pp.60-63.
8. Le Hoang Thai, Tran Son Hai (2009), *Phat Trien Mo Hinh Lien Ket Mang Ap Dung Cho Bai Toan Phan Lop Anh*, Ky yeu hoi thao - Mot so van de chon loc ve Cong Nghe Thong Tin Quoc Gia, pp. 327-340.
9. Le Hoang Thai, Nguyen Do Thai Nguyen and Tran Son Hai (Oct 2011), *A Facial Expression Classification System Integrating Canny, Principal Component Analysis and Artificial Neural Network*, International Journal of Machine Learning and Computing, Vol. 1, No. 4, pp 388-393.
10. Le Hoang Thai, Tran Son Hai, Nguyen Thanh Thuy (May 2012), *Image Classification using Support Vector Machine and Artificial Neural Network*, I.J. Information Technology and Computer Science, Vol. 5, pp. 32-38, DOI: 10.5815/ijitcs.2012.05.05
11. M.M. Spadotto, P.R. Aguiar, C.C.P. Souza, E.C. Bianchi, and A.N. de Souza, *Classification of Burn Degrees in Grinding by Neural Nets*, Artificial Intelligence and Applications (AIA), pp. 595-140, 2008.
12. Lipo Wang. *Support Vector Machines: Theory and Applications*. Springer.
13. Xindong Wu, Vipin Kumar, et al. (2008), *The Top Ten Algorithms in Data Mining*, Springer.

---

**Giảng viên hướng dẫn** | **Học viên cao học**
---|---
PGS.TS Lê Hoàng Thái | Lê Quang Thái
