# Flowchart cho finetune_setA_findingaware.py

```mermaid
flowchart TD
    A[Start] --> B[Đọc file Excel chứa label]
    B --> C[Chuyển nhãn Khong/Co thành 0/1]
    C --> D[Quét thư mục ảnh]
    D --> E[Tạo mapping tên file -> đường dẫn ảnh]
    E --> F[Ghép ảnh vào dataframe]
    F --> G[Loại bỏ dòng không tìm thấy ảnh]
    G --> H[Tạo nhóm bệnh nhân từ tên file]
    H --> I[Chia dữ liệu thành Train / Val / Test]
    I --> J[Tạo Dataset từ ảnh và nhãn]
    J --> K[Tạo DataLoader]
    K --> L[Build model BioViL-T]
    L --> M[Load backbone finding-aware]
    M --> N[Freeze backbone, train head]
    N --> O[Train epoch]
    O --> P[Validate]
    P --> Q[Save checkpoint tốt nhất]
    Q --> R[Unfreeze backbone ở epoch sau]
    R --> S[Tiếp tục training]
    S --> T[Đánh giá trên test set]
    T --> U[Vẽ confusion matrix / ROC]
    U --> V[End]
```

## Ý nghĩa từng bước

- B: đọc dữ liệu từ Excel
- C: chuyển nhãn sang dạng số để train
- D-E: tìm và nối đường dẫn ảnh vào dataframe
- I: chia dữ liệu theo nhóm bệnh nhân để tránh leakage
- J-K: chuẩn bị dữ liệu cho PyTorch
- L-M: nạp backbone BioViL-T đã học trước
- N-S: train model theo 2 giai đoạn
- T-U: đánh giá hiệu suất bằng metric như ROC AUC, accuracy, sensitivity, specificity
