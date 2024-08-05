1. Cấu trúc core A3T AI:
    - Thư mục A3T-core: Lưu trữ source code liên quan đến AI (training, testing, inference, tools, etc.)
2. Cấu trúc thư mục A3T-core:
    - File app.py: sử dụng socket để nhận các yêu cầu từ service gửi đến, tiến hành các yêu cầu thực thi chạy mô hình AI trên các luồng khác nhau.
    - File train.py, test.py, inference.py: thực thi các yêu cầu tương ứng là huấn luyện, kiểm thử và suy luận mô hình. Được đặt trong 1 hàm có input và ouput tương ứng.
    - Có thể cấu hình thêm các file khác phục vụ mục đích hỗ trợ các hàm chính
4. Flow thêm socket:
    - Bước 1: Vào file app.py để nhận luồng socket
        @sio.on("<tên event>")
        async def <tên event>_listener(data):
            Thread(target=await <tên event>_thread(data)).start()

        async def <tên event>_thread(data):
        try:
            async def func_with_timeout():
                responseComparing = await <hàm được gọi để thực thi mô hình AI hoặc các hàm khác>(
                    //payload
                )
                await sio.emit(
                    f"<tên sự kiện lắng nghe (C)>",
                    json.dumps(
                        {
                            "responseComparing": responseComparing,
                            "labId": data["labId"],
                        }
                    ),
                )
                await sio.sleep(0.1)
            await asyncio.wait_for(func_with_timeout(), timeout=TIMEOUT_LIMIT)
        except asyncio.TimeoutError:
            await sio.emit(
                "<tên sự kiện lắng nghe (C)>",
                json.dumps(
                    {
                        "response": {
                            "message": "Timeout. Please try again later.",
                            "status": False
                        },
                        "labId": data["labId"],
                    }
                ),
            )
5. Flow chi tiết về A3T-core
- Định nghĩa các file train, test, inference trong các file python tương ứng
- Các file train, test, inference thì nên đặt ra các file riêng, và xử lý 1 hàm, có đầu vào là các input tương ứng tùy thuộc vào mô hình và xác định bài toán. Các biến chung cần có bao gồm:
    + train.py: data_dir - thư mục dữ liệu train, learning_rate, epochs, batch_size, val_size, num_neurons - số nơ ron từng layer trong CNN, num_layers - số lớp ẩn, backbone - cấu trúc mô hình, model_type - loại mô hình, labId - mã bài thí nghiệm
    + test.py: test_data_dir - thư mục dữ liệu test, labId, ckpt_number - mô hình ở epoch thứ mấy được chọn để test, model_type, num_neurons, num_layers, backbone, sample_model_dir - mô hình mẫu để đối sánh với mô hình đã huấn luyện
    + inference.py: sample - dữ liệu mẫu để infer, labId, ckpt_number, model_type, num_neurons, num_layers, backbone, sample_model_dir=''
- Đối với phần train thì có thể cấu hình để trả ra kết quả theo từng epoch, sử dụng yield trong vòng lặp for...loop để sau mỗi vòng lặp sẽ trả ra kêt quả:
    Ví dụ: for res_per_epoch in train_output:
                yield res_per_epoch
- Thêm try...catch ở các func để tránh treo hệ thống.
- Cấu trúc file app.py làm theo code mẫu, chỉ thay đổi ở các phần socket gọi đến các mô hình AI tương ứng và thay đổi các biến hệ thống:
    TIMEOUT_LIMIT = 60 * 60 * 24  # Thời gian chờ đợi để đưa ra kết quả mô hình AI, nếu vượt quá thời gian sẽ break
    SOCKET_BACKEND_URL = "http://url-phishing-service:12007" # đường dẫn kết nối socket tới A3T-service
    PORT = 12009 # cổng dể đẩy hệ thống chạy trên môi trường test và product