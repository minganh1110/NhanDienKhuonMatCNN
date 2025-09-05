# Nhận diện khuôn mặt bằng CNN + SVM

Dự án này xây dựng hệ thống **nhận diện khuôn mặt** gồm 3 phần chính:  
1. **Thu thập dữ liệu** (ảnh khuôn mặt).  
2. **Huấn luyện mô hình** (CNN → trích xuất đặc trưng → SVM phân loại).  
3. **Ứng dụng giao diện** để dự đoán khuôn mặt từ webcam hoặc ảnh đầu vào.  

---

## Cấu trúc dự án
├── dataset/ # Thư mục chứa ảnh khuôn mặt (theo từng người)

├── face_buffalo_l_svm.pkl # Mô hình SVM đã train, dùng để dự đoán

├── giaodien.py # File giao diện chạy chương trình nhận diện

├── NhanDienKhuonMatCNN2.ipynb# Notebook huấn luyện CNN

├── pipeline.txt # Mô tả pipeline hệ thống

├── thuthapkhuonmat.py # Script thu thập dữ liệu ảnh khuôn mặt


---

## Pipeline xử lý
1. **Thu thập ảnh**:  
   - Chạy `thuthapkhuonmat.py` để chụp ảnh từ webcam và lưu vào thư mục `dataset/`.
   - Ảnh được lưu theo tên id ví dụ : 1_50.jpg là id = 1 và bức ảnh thứ 50
     
2. **Data Processing (Tiền xử lý dữ liệu)**  
   - Chuyển ảnh về kích thước chuẩn (thường 112×112).  
   - Chuẩn hóa giá trị pixel về khoảng `[0,1]`.  
   - Phát hiện và cắt khuôn mặt, loại bỏ ảnh không có khuôn mặt.  

3. **Data Augmentation (Tăng cường dữ liệu)**  
   - Lật ngang khuôn mặt.  
   - Xoay nhẹ ±10 độ.  
   - Thay đổi độ sáng, độ tương phản.  
   - Thêm nhiễu nhẹ (noise).  
   → Mục đích: tăng tính đa dạng dữ liệu, giúp mô hình nhận diện tốt hơn trong điều kiện thực tế.  

4. **Trích xuất embedding**  
   - Dùng mô hình `buffalo_l` để biến mỗi ảnh thành vector 512 chiều.  
   - Đây chính là đặc trưng khuôn mặt, không phụ thuộc vào ánh sáng hay góc chụp nhiều.  

5. **Huấn luyện SVM**  
   - Lấy embedding và nhãn (ID người) → train mô hình SVM.  
   - Lưu kết quả thành `face_buffalo_l_svm.pkl`.  

6. **Nhận diện khuôn mặt**  
   - Chạy `giaodien.py`.  
   - Mở webcam → phát hiện khuôn mặt → trích embedding bằng `buffalo_l` → phân loại bằng SVM → hiển thị tên người.  

---


