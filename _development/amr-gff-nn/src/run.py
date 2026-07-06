def main():
    # 1. 預加載數據與環境 (只做一次)
    loader = build_loaders(config.data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    while True:
        print("\n=== GatedFusionFormer 研究控制台 ===")
        print(" [1] 訓練新模型 (Train)")
        print(" [2] 基礎評估 (Accuracy vs SNR)")
        print(" [3] 深度診斷 (Confusion Matrix / Gating Weights)")
        print(" [4] 效能分析 (Latency / FLOPs)")
        print(" [q] 退出實驗室")
        
        step = input("\n請輸入實驗步驟 [1-4/q]: ").strip().lower()
        
        if step == '1':
            m_name = input("選擇模型 [mcl_dnn/sa_cnn/iq_former/gff_v3]: ")
            model = build_model(m_name, config[m_name]).to(device)
            train_runner(model, loader)
        elif step == '2':
            # 直接使用內存中已加載的 loader 進行推理
            run_snr_accuracy_test(current_model, loader)
        elif step == 'q':
            break