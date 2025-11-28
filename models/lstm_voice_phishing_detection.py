"""
LSTM 기반 음성 피싱 탐지 모델
Voice Phishing Detection using LSTM Deep Learning Model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')


class VoicePhishingLSTMDetector:
    """
    LSTM을 이용한 음성 피싱 탐지 시스템
    
    특징:
    - FastText 임베딩을 이용한 토큰 벡터화
    - LSTM 레이어로 시계열 패턴 학습
    - Bidirectional LSTM으로 양방향 문맥 학습
    - Attention 메커니즘 (선택사항)
    """
    
    def __init__(self, embedding_dim=300, max_sequence_length=200, verbose=True):
        """
        초기화
        
        Args:
            embedding_dim: 임베딩 차원 (FastText 기본값: 300)
            max_sequence_length: 최대 시퀀스 길이
            verbose: 로그 출력 여부
        """
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.verbose = verbose
        self.model = None
        self.history = None
        
    def tokens_to_vectors(self, token_list, fasttext_model):
        """
        토큰 리스트를 FastText 임베딩으로 변환
        
        Args:
            token_list: 토큰화된 텍스트 리스트
            fasttext_model: FastText 모델
            
        Returns:
            numpy 배열 (샘플 수, 최대 길이, 임베딩 차원)
        """
        vectors = []
        
        for tokens in token_list:
            token_vectors = []
            
            # 각 토큰을 벡터로 변환
            for token in tokens:
                try:
                    vec = fasttext_model[token]
                    token_vectors.append(vec)
                except:
                    # 모델에 없는 토큰은 0 벡터로 대체
                    token_vectors.append(np.zeros(self.embedding_dim))
            
            if len(token_vectors) > 0:
                token_vectors = np.array(token_vectors)
            else:
                token_vectors = np.zeros((1, self.embedding_dim))
            
            vectors.append(token_vectors)
        
        # 모든 시퀀스를 같은 길이로 패딩/트렁케이션
        padded_vectors = []
        for vec in vectors:
            if len(vec) > self.max_sequence_length:
                # 트렁케이션
                padded_vectors.append(vec[:self.max_sequence_length])
            else:
                # 패딩
                pad_length = self.max_sequence_length - len(vec)
                padded_vec = np.vstack([vec, np.zeros((pad_length, vec.shape[1]))])
                padded_vectors.append(padded_vec)
        
        return np.array(padded_vectors)
    
    def build_model(self, model_type='lstm'):
        """
        모델 구축
        
        Args:
            model_type: 'lstm', 'bilstm', 또는 'bilstm_attention'
        """
        if model_type == 'lstm':
            self.model = self._build_lstm_model()
        elif model_type == 'bilstm':
            self.model = self._build_bilstm_model()
        elif model_type == 'bilstm_attention':
            self.model = self._build_bilstm_attention_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if self.verbose:
            print(f"\n{model_type.upper()} 모델 구축 완료")
            self.model.summary()
    
    def _build_lstm_model(self):
        """기본 LSTM 모델"""
        model = Sequential([
            layers.LSTM(64, input_shape=(self.max_sequence_length, self.embedding_dim),
                       return_sequences=True, dropout=0.2),
            layers.LSTM(32, return_sequences=False, dropout=0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Recall(), keras.metrics.Precision()]
        )
        return model
    
    def _build_bilstm_model(self):
        """양방향 LSTM 모델"""
        model = Sequential([
            layers.Bidirectional(
                layers.LSTM(64, input_shape=(self.max_sequence_length, self.embedding_dim),
                           return_sequences=True, dropout=0.2)
            ),
            layers.Bidirectional(
                layers.LSTM(32, return_sequences=False, dropout=0.2)
            ),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Recall(), keras.metrics.Precision()]
        )
        return model
    
    def _build_bilstm_attention_model(self):
        """Attention이 포함된 양방향 LSTM 모델"""
        inputs = keras.Input(shape=(self.max_sequence_length, self.embedding_dim))
        
        # 양방향 LSTM
        lstm = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2)
        )(inputs)
        
        # Attention 메커니즘
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(lstm, lstm)
        attention = layers.Add()([attention, lstm])  # Residual connection
        attention = layers.LayerNormalization()(attention)
        
        # 두 번째 LSTM
        lstm2 = layers.Bidirectional(
            layers.LSTM(32, return_sequences=False, dropout=0.2)
        )(attention)
        
        # Dense 레이어
        dense = layers.Dense(16, activation='relu')(lstm2)
        dense = layers.Dropout(0.3)(dense)
        outputs = layers.Dense(1, activation='sigmoid')(dense)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Recall(), keras.metrics.Precision()]
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        모델 훈련
        
        Args:
            X_train: 훈련 데이터 (벡터)
            y_train: 훈련 레이블 (0: 정상, 1: 피싱)
            X_val: 검증 데이터
            y_val: 검증 레이블
            epochs: 에포크 수
            batch_size: 배치 크기
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 if self.verbose else 0
        )
        
        if self.verbose:
            print("\n훈련 완료!")
    
    def evaluate(self, X_test, y_test):
        """
        모델 평가
        
        Args:
            X_test: 테스트 데이터
            y_test: 테스트 레이블
            
        Returns:
            평가 메트릭 딕셔너리
        """
        # 예측
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_test_flat = y_test.flatten() if isinstance(y_test, np.ndarray) else np.array(y_test)
        
        # 메트릭 계산
        metrics = {
            'accuracy': accuracy_score(y_test_flat, y_pred),
            'recall': recall_score(y_test_flat, y_pred),
            'precision': precision_score(y_test_flat, y_pred),
            'f1': f1_score(y_test_flat, y_pred),
            'roc_auc': roc_auc_score(y_test_flat, y_pred_prob)
        }
        
        if self.verbose:
            print("\n===== 모델 평가 결과 =====")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"F1-Score:  {metrics['f1']:.4f}")
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            # 혼동 행렬
            cm = confusion_matrix(y_test_flat, y_pred)
            print(f"\n혼동 행렬:\n{cm}")
        
        return metrics, y_pred_prob
    
    def plot_training_history(self):
        """훈련 히스토리 시각화"""
        if self.history is None:
            print("훈련 히스토리가 없습니다.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """모델 저장"""
        self.model.save(filepath)
        if self.verbose:
            print(f"모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath):
        """모델 로드"""
        self.model = keras.models.load_model(filepath)
        if self.verbose:
            print(f"모델이 {filepath}에서 로드되었습니다.")


def main():
    """메인 함수 - 사용 예시"""
    print("=" * 60)
    print("LSTM 기반 음성 피싱 탐지 모델")
    print("=" * 60)
    
    # 1. 모델 초기화
    detector = VoicePhishingLSTMDetector(
        embedding_dim=300,
        max_sequence_length=200,
        verbose=True
    )
    
    print("\n[Step 1] 모델 구축 중...")
    detector.build_model(model_type='bilstm')
    
    print("\n사용 방법:")
    print("1. detector.tokens_to_vectors(token_list, fasttext_model)")
    print("   - 토큰을 벡터로 변환")
    print("\n2. detector.train(X_train, y_train, X_val, y_val)")
    print("   - 모델 훈련")
    print("\n3. metrics, y_pred = detector.evaluate(X_test, y_test)")
    print("   - 모델 평가")
    print("\n4. detector.plot_training_history()")
    print("   - 훈련 히스토리 시각화")
    print("\n5. detector.save_model('model.h5')")
    print("   - 모델 저장")


if __name__ == "__main__":
    main()
