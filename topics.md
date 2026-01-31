1. Phát biểu bài toán (Problem Statement)
Tên đề tài gợi ý: Xây dựng hệ thống tự động hóa đánh giá biến đổi giọng nói tiền và hậu phẫu dựa trên phân tích đặc trưng âm học đa chiều (F0, Formants và Perturbation).
Bài toán:
Input (X): Tập các tín hiệu âm thanh thô S={s1,s2,s3,s4,s5}
 tương ứng với 5 nguyên âm kéo dài $/a, e, i, o, u/$ của cùng một bệnh nhân tại hai thời điểm T0 (trước mổ) và T1 (sau mổ).
Quá trình xử lý (P):
Trích xuất đặc trưng (Feature Extraction): Chuyển đổi tín hiệu miền thời gian sang miền tần số để trích xuất:
Nguồn âm (Source): Tần số cơ bản F0 và độ lệch chuẩn F0SD.
Bộ lọc (Filter): Tọa độ các đỉnh cộng hưởng F1,F2,F3.
Nhiễu loạn (Perturbation): Jitter, Shimmer, HNR và chỉ số Cepstral Peak Prominence (CPP).
Mô hình hóa không gian nguyên âm (Vowel Space Modeling): Áp dụng thuật toán tính toán diện tích đa giác lồi (Convex Hull Area) trên mặt phẳng F1-F2 để xác định chỉ số Vowel Space Area (VSA) và Vowel Articulation Index (VAI).
Phân tích biến đổi (Longitudinal Analysis): Tính toán vector sai lệch V=VT1-VT0 để định lượng mức độ thay đổi chức năng cấu âm và phát âm.
Output (Y): Hệ thống chẩn đoán hỗ trợ (Computer-Aided Diagnosis - CAD) đưa ra:
Biểu đồ so sánh trực quan (VSA map, Radar chart).
Phân loại mức độ bệnh lý giọng nói (Healthy vs. Pathological) với độ chính xác mục tiêu >90%

2. Dữ liệu nghiên cứu công khai (Public Datasets)
Để học viên có thể thực hiện luận văn ngay mà không cần chờ thu thập dữ liệu lâm sàng (vốn rất khó khăn về thủ tục đạo đức), bạn nên hướng dẫn học viên sử dụng các tập dữ liệu "vàng" sau:
A. Saarbrücken Voice Database (SVD) - Toàn diện nhất
Đây là tập dữ liệu chuẩn cho phép thực hiện đúng yêu cầu về các nguyên âm khác nhau.
Nội dung: Chứa bản ghi âm của hơn 2.000 người (khỏe mạnh và bệnh lý với 71 loại rối loạn thanh quản).
Đặc trưng: Bao gồm các nguyên âm kéo dài /a, i, u/ được ghi ở nhiều cao độ.
Link tải:(https://stimmdb.coli.uni-saarland.de/).
B. Advanced Voice Function Assessment Database (AVFAD) - Giàu tham số
Rất phù hợp để so sánh vì nó cung cấp sẵn các thông số trích xuất từ phần mềm chuyên dụng Praat.
Nội dung: 709 đối tượng phát âm nguyên âm /a, e, o/.
Đặc trưng: Cung cấp sẵn 19 tham số âm học (F0, Jitter, Shimmer, HNR,...) cho mỗi mẫu.
Link tải:(https://www.researchgate.net/publication/319876992_The_Advanced_Voice_Function_Assessment_Databases_AVFAD_Tools_for_voice_clinicians_and_speech_engineering_research)
C. VOICED Database (trên PhysioNet) - Dữ liệu thực tế từ Smartphone
Phù hợp nếu học viên muốn làm phần mềm chạy trên thiết bị di động.
Nội dung: 208 mẫu giọng nói (150 bệnh lý, 58 khỏe mạnh) thu âm bằng Samsung Galaxy S4.
Đặc trưng: Tập trung vào nguyên âm /a/ kéo dài 5 giây.
Link tải:(https://physionet.org/content/voiced/1.0.0/) hoặc Kaggle
D. FEMH Voice Dataset - Dành cho các bài toán thử thách
Nội dung: 2.000 cá nhân phát âm nguyên âm /a/, bao gồm cả các ca ung thư thanh quản và liệt dây thanh.
Link tải:(https://www.researchgate.net/publication/330629543_FEMH_Voice_Data_Challenge_Voice_disorder_Detection_and_Classification_using_Acoustic_Descriptors).
3. Gợi ý công cụ triển khai cho học viên
Học viên nên sử dụng ngôn ngữ Python kết hợp với:
Parselmouth: Để gọi thư viện Praat trích xuất F0,F1,F2,F3 chính xác theo tiêu chuẩn y khoa.
Librosa: Để tiền xử lý và trích xuất các đặc trưng phổ (Mel-spectrogram).
Scikit-learn/PyTorch: Để xây dựng mô hình phân loại mức độ thay đổi dựa trên các vector đặc trưng thu được.

