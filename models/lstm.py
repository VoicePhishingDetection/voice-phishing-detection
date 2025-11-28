"""
Training helper script for voice-phishing LSTM models.

This module contains helper functions to prepare data, train multiple
models and perform ensemble prediction. Hyperparameters are imported
from `config.model` so tuning is centralized.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from models.lstm_voice_phishing_detection import VoicePhishingLSTMDetector
import matplotlib.pyplot as plt
from config.model import EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, EPOCHS, BATCH_SIZE, THRESHOLD


def prepare_data_for_training(voice_phishing_tokens, normal_tokens, fasttext_model,
                              test_size=0.2, val_size=0.2):
    """Prepare vectors and split into train/val/test sets.

    Returns X_train, X_val, X_test, y_train, y_val, y_test, detector
    """
    detector = VoicePhishingLSTMDetector(
        embedding_dim=EMBEDDING_DIM,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        verbose=True
    )

    print("\n[데이터 전처리] 음성 피싱 데이터 벡터화 중...")
    voice_phishing_vectors = detector.tokens_to_vectors(voice_phishing_tokens, fasttext_model)
    print(f"음성 피싱 벡터 형태: {voice_phishing_vectors.shape}")

    print("[데이터 전처리] 정상 데이터 벡터화 중...")
    normal_vectors = detector.tokens_to_vectors(normal_tokens, fasttext_model)
    print(f"정상 벡터 형태: {normal_vectors.shape}")

    y_phishing = np.ones(len(voice_phishing_vectors))
    y_normal = np.zeros(len(normal_vectors))

    X = np.vstack([voice_phishing_vectors, normal_vectors])
    y = np.hstack([y_phishing, y_normal])

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, detector


def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train multiple model variants and compare results."""
    model_types = ['lstm', 'bilstm', 'bilstm_attention']
    results = {}

    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"모델 타입: {model_type.upper()}")
        print(f"{'='*60}")

        detector = VoicePhishingLSTMDetector(
            embedding_dim=EMBEDDING_DIM,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            verbose=True
        )

        detector.build_model(model_type=model_type)

        print(f"\n[Step 2] 모델 훈련 중...")
        detector.train(
            X_train, y_train,
            X_val, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        detector.plot_training_history()

        print(f"\n[Step 3] 모델 평가 중...")
        metrics, y_pred_prob = detector.evaluate(X_test, y_test)

        results[model_type] = {
            'detector': detector,
            'metrics': metrics,
            'y_pred_prob': y_pred_prob
        }

        detector.save_model(f'voice_phishing_lstm_{model_type}.h5')

    # Compare
    print(f"\n{'='*60}")
    print("모델 성능 비교")
    print(f"{'='*60}")

    comparison_data = {
        'Model': [], 'Accuracy': [], 'Recall': [], 'Precision': [], 'F1-Score': [], 'ROC-AUC': []
    }

    for model_type, result in results.items():
        metrics = result['metrics']
        comparison_data['Model'].append(model_type.upper())
        comparison_data['Accuracy'].append(f"{metrics['accuracy']:.4f}")
        comparison_data['Recall'].append(f"{metrics['recall']:.4f}")
        comparison_data['Precision'].append(f"{metrics['precision']:.4f}")
        comparison_data['F1-Score'].append(f"{metrics['f1']:.4f}")
        comparison_data['ROC-AUC'].append(f"{metrics['roc_auc']:.4f}")

    print(f"\n{'Model':<20} {'Accuracy':<12} {'Recall':<12} {'Precision':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 80)
    for i in range(len(comparison_data['Model'])):
        print(
            f"{comparison_data['Model'][i]:<20} "
            f"{comparison_data['Accuracy'][i]:<12} "
            f"{comparison_data['Recall'][i]:<12} "
            f"{comparison_data['Precision'][i]:<12} "
            f"{comparison_data['F1-Score'][i]:<12} "
            f"{comparison_data['ROC-AUC'][i]:<12}"
        )

    return results


def ensemble_prediction(results, X_test, y_test):
    """Ensemble prediction by averaging model probabilities."""
    print(f"\n{'='*60}")
    print("앙상블 모델 (3개 모델의 평균)")
    print(f"{'='*60}\n")

    ensemble_pred_prob = np.zeros_like(results['lstm']['y_pred_prob'])
    for model_type in ['lstm', 'bilstm', 'bilstm_attention']:
        ensemble_pred_prob += results[model_type]['y_pred_prob']

    ensemble_pred_prob /= 3
    ensemble_pred = (ensemble_pred_prob > THRESHOLD).astype(int).flatten()

    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

    y_test_flat = y_test.flatten()

    print(f"Accuracy:  {accuracy_score(y_test_flat, ensemble_pred):.4f}")
    print(f"Recall:    {recall_score(y_test_flat, ensemble_pred):.4f}")
    print(f"Precision: {precision_score(y_test_flat, ensemble_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test_flat, ensemble_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test_flat, ensemble_pred_prob):.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("LSTM 음성 피싱 탐지 - 훈련 및 평가")
    print("=" * 60)

    print("\n주의: 이 스크립트를 실행하려면 다음이 필요합니다:")
    print("1. voice_phishing_train_token - 음성 피싱 훈련 토큰")
    print("2. normal_train_token - 정상 훈련 토큰")
    print("3. ko_model - FastText 한국어 모델")
    print("\n다음과 같이 사용하세요:")
    print('''
    from models.lstm import prepare_data_for_training, train_and_evaluate_models
    
    # 데이터 준비
    X_train, X_val, X_test, y_train, y_val, y_test, detector = prepare_data_for_training(
        voice_phishing_train_token,
        normal_train_token,
        ko_model
    )
    
    # 모델 훈련 및 평가
    results = train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # 앙상블 예측
    ensemble_prediction(results, X_test, y_test)
    ''')
