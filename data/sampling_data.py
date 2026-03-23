import pandas as pd
import os

input_file = 'data.csv'  # Thay bằng tên file gốc của bạn
output_file = 'student_academic_performance_1M.csv'   # Tên file mới sẽ được tạo ra

print("Đang đọc file gốc...")
df = pd.read_csv(input_file)
print(f"Số dòng ban đầu: {len(df):,}")

# Lấy mẫu 20% dữ liệu (0.2). Bạn có thể thay đổi con số này.
# Ví dụ: 0.15 là 15%, 0.3 là 30%...
ty_le = 0.2 
df_sampled = df.sample(frac=ty_le, random_state=42)

print(f"Số dòng sau khi cắt giảm: {len(df_sampled):,}")
print("Đang lưu ra file mới...")

# Lưu thành file .csv chuẩn, không nén, bỏ cột index
df_sampled.to_csv(output_file, index=False)

# Kiểm tra lại dung lượng file mới
new_size = os.path.getsize(output_file) / (1024 * 1024)
print("-" * 30)
print(f"✅ Dung lượng file mới: {new_size:.2f} MB")

if new_size < 100:
    print("🎉 Tuyệt vời! File đã đủ nhỏ, bạn có thể push thẳng lên GitHub.")
else:
    print("⚠️ Cảnh báo: File vẫn lớn hơn 100MB. Bạn hãy sửa biến `ty_le` thành 0.15 hoặc 0.1 rồi chạy lại code nhé.")