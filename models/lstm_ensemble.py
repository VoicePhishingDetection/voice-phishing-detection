"""
통계 기반 방법 + LSTM 모델 앙상블
Statistical Methods + LSTM Ensemble for Voice Phishing Detection
"""

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


class EnsembleVoicePhishingDetector:
    """
    기존 통계 기반 방법과 LSTM을 결합한 앙상블 모델
    
    방법:
    - Statistical Score: 기존 편차 + EWMA 기반 점수 (0~1 범위)
    - LSTM Score: LSTM 모델의 예측 확률 (0~1 범위)
    - Final Score: weighted_statistical * statistical_score + weighted_lstm * lstm_score
    """
    
    def __init__(self, statistical_weight=0.4, lstm_weight=0.6, verbose=True):
        """
        초기화
        
        Args:
            statistical_weight: 통계 방법 가중치 (기본: 0.4)
            lstm_weight: LSTM 가중치 (기본: 0.6)
            verbose: 로그 출력 여부
        """
        self.statistical_weight = statistical_weight
        self.lstm_weight = lstm_weight
        self.verbose = verbose
        
        assert abs((statistical_weight + lstm_weight) - 1.0) < 1e-6, \
            "가중치 합이 1이어야 합니다"
    
    def normalize_statistical_scores(self, statistical_scores, method='minmax'):
        """
        통계 점수를 정규화 (0~1 범위)
        
        Args:
            statistical_scores: 원본 통계 점수 (각 샘플의 값)
            method: 'minmax' 또는 'sigmoid'
            
        Returns:
            정규화된 점수 (0~1 범위)
        """
        if method == 'minmax':
            min_val = np.min(statistical_scores)
            max_val = np.max(statistical_scores)
            if max_val - min_val == 0:
                return np.ones_like(statistical_scores) * 0.5
            normalized = (statistical_scores - min_val) / (max_val - min_val)
        elif method == 'sigmoid':
            # 평균과 표준편차를 이용한 sigmoid
            mean = np.mean(statistical_scores)
            std = np.std(statistical_scores)
            if std == 0:
                return np.ones_like(statistical_scores) * 0.5
            z_scores = (statistical_scores - mean) / std
            normalized = 1 / (1 + np.exp(-z_scores))
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def ensemble_predict(self, statistical_scores, lstm_pred_probs, threshold=0.5):
        """
        앙상블 예측
        
        Args:
            statistical_scores: 통계 방법 점수 배열 (정규화 전)
            lstm_pred_probs: LSTM 예측 확률 배열 (이미 0~1 범위)
            threshold: 분류 임계값
            
        Returns:
            ensemble_scores: 앙상블 점수 (0~1 범위)
            predictions: 최종 예측 (0 또는 1)
        """
        # 통계 점수 정규화
        normalized_statistical = self.normalize_statistical_scores(
            statistical_scores, method='sigmoid'
        )
        
        # LSTM 점수 정규화 (이미 정규화되어 있으므로 그대로 사용)
        normalized_lstm = lstm_pred_probs.flatten()
        
        # 앙상블 계산
        ensemble_scores = (
            self.statistical_weight * normalized_statistical +
            self.lstm_weight * normalized_lstm
        )
        
        # 최종 예측
        predictions = (ensemble_scores > threshold).astype(int)
        
        return ensemble_scores, predictions
    
    def evaluate_ensemble(self, statistical_scores, lstm_pred_probs, y_true, threshold=0.5):
        """
        앙상블 모델 평가
        
        Args:
            statistical_scores: 통계 방법 점수
            lstm_pred_probs: LSTM 예측 확률
            y_true: 정답 레이블
            threshold: 분류 임계값
            
        Returns:
            metrics 딕셔너리
        """
        ensemble_scores, predictions = self.ensemble_predict(
            statistical_scores, lstm_pred_probs, threshold
        )
        
        y_true_flat = y_true.flatten() if isinstance(y_true, np.ndarray) else np.array(y_true)
        
        metrics = {
            'accuracy': accuracy_score(y_true_flat, predictions),
            'recall': recall_score(y_true_flat, predictions),
            'precision': precision_score(y_true_flat, predictions),
            'f1': f1_score(y_true_flat, predictions),
            'roc_auc': roc_auc_score(y_true_flat, ensemble_scores)
        }
        
        if self.verbose:
            print("\n===== 앙상블 모델 평가 결과 =====")
            print(f"가중치: Statistical={self.statistical_weight}, LSTM={self.lstm_weight}")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"F1-Score:  {metrics['f1']:.4f}")
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics, ensemble_scores
    
    def find_optimal_threshold(self, statistical_scores, lstm_pred_probs, y_true):
        """
        최적 임계값 찾기 (F1-score 기준)
        
        Args:
            statistical_scores: 통계 방법 점수
            lstm_pred_probs: LSTM 예측 확률
            y_true: 정답 레이블
            
        Returns:
            optimal_threshold: 최적 임계값
            metrics_at_threshold: 각 임계값에서의 메트릭스
        """
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_f1 = 0
        optimal_threshold = 0.5
        metrics_at_threshold = {}
        
        for threshold in thresholds:
            _, preds = self.ensemble_predict(statistical_scores, lstm_pred_probs, threshold)
            y_true_flat = y_true.flatten()
            
            f1 = f1_score(y_true_flat, preds)
            metrics_at_threshold[threshold] = f1
            
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = threshold
        
        if self.verbose:
            print(f"\n최적 임계값: {optimal_threshold:.2f} (F1-Score: {best_f1:.4f})")
        
        return optimal_threshold, metrics_at_threshold
    
    def compare_methods(self, statistical_scores, lstm_pred_probs, y_true):
        """
        개별 방법과 앙상블 방법 비교
        
        Args:
            statistical_scores: 통계 방법 점수
            lstm_pred_probs: LSTM 예측 확률
            y_true: 정답 레이블
        """
        y_true_flat = y_true.flatten()
        
        print("\n" + "="*70)
        print("방법별 성능 비교")
        print("="*70)
        
        # 1. 통계 방법만 사용
        normalized_stat = self.normalize_statistical_scores(statistical_scores)
        stat_pred = (normalized_stat > 0.5).astype(int)
        
        print("\n[1] 통계 방법만 사용")
        print(f"Accuracy:  {accuracy_score(y_true_flat, stat_pred):.4f}")
        print(f"Recall:    {recall_score(y_true_flat, stat_pred):.4f}")
        print(f"Precision: {precision_score(y_true_flat, stat_pred):.4f}")
        print(f"F1-Score:  {f1_score(y_true_flat, stat_pred):.4f}")
        
        # 2. LSTM만 사용
        lstm_pred = (lstm_pred_probs.flatten() > 0.5).astype(int)
        
        print("\n[2] LSTM만 사용")
        print(f"Accuracy:  {accuracy_score(y_true_flat, lstm_pred):.4f}")
        print(f"Recall:    {recall_score(y_true_flat, lstm_pred):.4f}")
        print(f"Precision: {precision_score(y_true_flat, lstm_pred):.4f}")
        print(f"F1-Score:  {f1_score(y_true_flat, lstm_pred):.4f}")
        
        # 3. 앙상블 방법
        _, ensemble_pred = self.ensemble_predict(statistical_scores, lstm_pred_probs)
        
        print("\n[3] 앙상블 방법 (Statistical=0.4, LSTM=0.6)")
        print(f"Accuracy:  {accuracy_score(y_true_flat, ensemble_pred):.4f}")
        print(f"Recall:    {recall_score(y_true_flat, ensemble_pred):.4f}")
        print(f"Precision: {precision_score(y_true_flat, ensemble_pred):.4f}")
        print(f"F1-Score:  {f1_score(y_true_flat, ensemble_pred):.4f}")
    
    def plot_ensemble_scores(self, statistical_scores, lstm_pred_probs, y_true):
        """
        앙상블 점수 시각화
        
        Args:
            statistical_scores: 통계 방법 점수
            lstm_pred_probs: LSTM 예측 확률
            y_true: 정답 레이블
        """
        normalized_stat = self.normalize_statistical_scores(statistical_scores)
        normalized_lstm = lstm_pred_probs.flatten()
        ensemble_scores, _ = self.ensemble_predict(statistical_scores, lstm_pred_probs)
        
        y_true_flat = y_true.flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 통계 점수
        axes[0, 0].hist(normalized_stat[y_true_flat == 0], bins=30, alpha=0.6, label='Normal', color='blue')
        axes[0, 0].hist(normalized_stat[y_true_flat == 1], bins=30, alpha=0.6, label='Phishing', color='red')
        axes[0, 0].set_title('Statistical Scores')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].legend()
        
        # LSTM 점수
        axes[0, 1].hist(normalized_lstm[y_true_flat == 0], bins=30, alpha=0.6, label='Normal', color='blue')
        axes[0, 1].hist(normalized_lstm[y_true_flat == 1], bins=30, alpha=0.6, label='Phishing', color='red')
        axes[0, 1].set_title('LSTM Scores')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].legend()
        
        # 앙상블 점수
        axes[1, 0].hist(ensemble_scores[y_true_flat == 0], bins=30, alpha=0.6, label='Normal', color='blue')
        axes[1, 0].hist(ensemble_scores[y_true_flat == 1], bins=30, alpha=0.6, label='Phishing', color='red')
        axes[1, 0].set_title('Ensemble Scores')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].axvline(x=0.5, color='green', linestyle='--', label='Threshold=0.5')
        axes[1, 0].legend()
        
        # 산점도
        axes[1, 1].scatter(normalized_stat[y_true_flat == 0], normalized_lstm[y_true_flat == 0],
                          alpha=0.5, label='Normal', color='blue', s=20)
        axes[1, 1].scatter(normalized_stat[y_true_flat == 1], normalized_lstm[y_true_flat == 1],
                          alpha=0.5, label='Phishing', color='red', s=20)
        axes[1, 1].set_title('Statistical vs LSTM Scores')
        axes[1, 1].set_xlabel('Statistical Scores')
        axes[1, 1].set_ylabel('LSTM Scores')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("앙상블 음성 피싱 탐지 모델")
    print("=" * 70)
    
    print("\n사용 방법:")
    print("""
    from lstm_ensemble import EnsembleVoicePhishingDetector
    
    # 앙상블 모델 초기화
    ensemble = EnsembleVoicePhishingDetector(
        statistical_weight=0.4,
        lstm_weight=0.6,
        verbose=True
    )
    
    # 1. 앙상블 예측
    scores, preds = ensemble.ensemble_predict(
        statistical_scores,  # 기존 통계 방법 점수
        lstm_pred_probs,     # LSTM 예측 확률
        threshold=0.5
    )
    
    # 2. 모델 평가
    metrics, scores = ensemble.evaluate_ensemble(
        statistical_scores,
        lstm_pred_probs,
        y_test
    )
    
    # 3. 최적 임계값 찾기
    optimal_threshold, metrics = ensemble.find_optimal_threshold(
        statistical_scores,
        lstm_pred_probs,
        y_test
    )
    
    # 4. 방법 비교
    ensemble.compare_methods(statistical_scores, lstm_pred_probs, y_test)
    
    # 5. 시각화
    ensemble.plot_ensemble_scores(statistical_scores, lstm_pred_probs, y_test)
    """)
