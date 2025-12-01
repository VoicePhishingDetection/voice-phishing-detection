import numpy as np
import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# TensorFlow/Keras 임포트
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model


# ============================================================
# SpamClassifier 클래스 정의
# 모델 구축, 학습, 예측 로직을 담당합니다.
# ============================================================

class SpamClassifier:
    """LSTM 기반 스팸/정상 메일 분류 모델 클래스"""

    def __init__(self, max_words=10000, max_len=100, threshold=0.6):
        self.max_words = max_words
        self.max_len = max_len
        self.threshold = threshold
        self.tokenizer = None
        self.model = None

    @staticmethod
    def load_assets(base_name='spam_detector'):
        """저장된 모델과 토크나이저를 불러옵니다."""
        model_path = f'{base_name}_lstm.keras'
        tokenizer_path = f'{base_name}_tokenizer.pkl'
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            try:
                model = load_model(model_path)
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
                print(f"✓ 모델 로드: {model_path}")
                print(f"✓ 토크나이저 로드: {tokenizer_path}")
                return model, tokenizer
            except Exception as e:
                print(f"⚠️ 저장된 모델 로드 실패: {e}")
                return None, None
        else:
            return None, None

    def build_model(self):
        """LSTM 모델 구조를 구축합니다."""
        self.model = Sequential([
            # 텍스트 임베딩 레이어
            Embedding(input_dim=self.max_words, output_dim=128, input_length=self.max_len),
            # 양방향 LSTM 레이어 1: 문맥 정보 파악
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.5),
            # 양방향 LSTM 레이어 2: 최종 특징 추출
            Bidirectional(LSTM(32)),
            Dropout(0.5),
            # 밀집 연결 레이어
            Dense(64, activation='relu'),
            Dropout(0.5),
            # 출력 레이어: 스팸 확률(0~1)을 예측
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("✓ 모델 구축 완료")
        self.model.summary()

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """모델을 학습시킵니다."""

        # 1. 토크나이저 구축 및 훈련 데이터에 맞게 적용
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)

        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len, padding='post', truncating='post')

        # Early Stopping 설정
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        print("\n[모델 학습 시작]")
        history = self.model.fit(
            X_train_pad, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        print("✓ 모델 학습 완료")
        return history

    def predict_email(self, text):
        """단일 이메일 텍스트를 입력받아 스팸 여부를 예측합니다."""
        if self.tokenizer is None or self.model is None:
            raise ValueError("모델이 훈련되지 않았거나 로드되지 않았습니다.")

        processed = self._preprocess_text(text)
        sequence = self.tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')

        # 예측 확률 계산
        prediction = self.model.predict(padded, verbose=0)[0][0]

        # 임계값을 사용하여 최종 레이블 결정
        label = "스팸 (Spam)" if prediction > self.threshold else "정상 (Ham)"
        # 신뢰도: 예측된 확률 또는 1에서 뺀 값 (항상 50% 이상)
        confidence = prediction if prediction > 0.5 else 1 - prediction

        return label, confidence

    def evaluate(self, X_test, y_test):
        """테스트 데이터로 모델 성능을 평가합니다."""
        if self.tokenizer is None or self.model is None:
            raise ValueError("모델이 훈련되지 않았거나 로드되지 않았습니다.")

        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len, padding='post', truncating='post')

        y_pred_prob = self.model.predict(X_test_pad)
        y_pred = (y_pred_prob > 0.5).astype(int)

        print("\n[모델 평가 결과]")
        print("분류 보고서:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

        # 혼동 행렬 시각화용 데이터 반환
        return confusion_matrix(y_test, y_pred)

    def save_assets(self, base_name='spam_detector'):
        """모델과 토크나이저를 저장합니다."""
        self.model.save(f'{base_name}_lstm.keras')
        with open(f'{base_name}_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"\n✓ 모델 저장: {base_name}_lstm.keras")
        print(f"✓ 토크나이저 저장: {base_name}_tokenizer.pkl")

    @staticmethod
    def _preprocess_text(text):
        """텍스트 전처리 (소문자, 특수문자 제거, 공백 정리)"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# ============================================================
# 메인 실행 로직
# ============================================================

def main():
    print("=" * 60)
    print("스팸 메일 탐지 LSTM 프로젝트 (클래스 기반)")
    print("=" * 60)

    DATASET_FILE = 'spam.csv'
    RANDOM_STATE = 42

    # 모델 초기화
    classifier = SpamClassifier(threshold=0.6)
    
    # 저장된 모델 확인 및 로드
    print("\n[0단계] 저장된 모델 확인...")
    model, tokenizer = SpamClassifier.load_assets()
    
    if model is not None and tokenizer is not None:
        # 저장된 모델이 있으면 로드하고 바로 예측으로 이동
        print("\n✓ 저장된 모델을 사용합니다.")
        classifier.model = model
        classifier.tokenizer = tokenizer
        
        print("\n" + "=" * 60)
        print("사용자 입력 실시간 예측 테스트")
        print("  종료하려면 'exit' 또는 'quit'를 입력하세요.")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n> 이메일 문장을 입력하세요: ")

                if user_input.lower() in ['exit', 'quit']:
                    break

                if not user_input.strip():
                    print("❗ 입력된 문장이 없습니다. 다시 입력해 주세요.")
                    continue

                label, confidence = classifier.predict_email(user_input)

                print("-" * 30)
                print(f"**이메일:** {user_input}")
                print(f"**예측 결과:** {label}")
                print(f"**신뢰도:** {confidence:.2%}")
                print("-" * 30)

            except Exception as e:
                print(f"⚠️ 오류가 발생했습니다: {e}")
                break
        
        return

    # 저장된 모델이 없으면 새로 학습
    print("\n⚠️ 저장된 모델이 없습니다. 새로 학습합니다.\n")
    print(f"\n[1단계] 데이터셋 로드 및 정리 ({DATASET_FILE})...")
    try:
        df = pd.read_csv(DATASET_FILE, encoding='latin-1', usecols=['v1', 'v2'])
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        if df.shape[1] > 2: df = df.iloc[:, :2]

        emails = df['text'].tolist()
        labels = df['label'].tolist()

        print(f"✓ 데이터 로드 완료. 총 {len(emails)}개 문장")

    except FileNotFoundError:
        print(f"⚠️ 오류: 파일 '{DATASET_FILE}'을 찾을 수 없습니다. 파일 이름을 확인해 주세요.")
        return
    except Exception as e:
        print(f"⚠️ 오류 발생: 데이터 로드 및 정리 실패 - {e}")
        return

    # 2. 데이터 분할
    print("\n[2단계] 데이터 분할 및 전처리...")
    processed_emails = [SpamClassifier._preprocess_text(email) for email in emails]
    X_train, X_test, y_train, y_test = train_test_split(
        processed_emails, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(f"✓ 학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")

    # 3. 모델 초기화 및 구축 (임계값 0.6 적용)
    print("\n[3단계] 모델 초기화 및 구축...")
    classifier = SpamClassifier(threshold=0.6)
    classifier.build_model()

    # 4. 모델 학습
    history = classifier.train(X_train, y_train)

    # 5. 모델 평가
    cm = classifier.evaluate(X_test, y_test)

    # 6. 학습 과정 그래프 (정확도 및 손실)
    print("\n[6단계] 학습 과정 그래프 출력...")

    # 정확도 그래프
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.suptitle('Model Training History', fontsize=16, fontweight='bold')
    plt.show()

    # 7. 혼동 행렬 시각화
    print("\n[7단계] 혼동 행렬 그래프 출력...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # 8. 사용자 입력 실시간 예측
    print("\n" + "=" * 60)
    print("사용자 입력 실시간 예측 테스트")
    print("  종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> 이메일 문장을 입력하세요: ")

            if user_input.lower() in ['exit', 'quit']:
                break

            if not user_input.strip():
                print("❗ 입력된 문장이 없습니다. 다시 입력해 주세요.")
                continue

            label, confidence = classifier.predict_email(user_input)

            print("-" * 30)
            print(f"**이메일:** {user_input}")
            print(f"**예측 결과:** {label}")
            print(f"**신뢰도:** {confidence:.2%}")
            print("-" * 30)

        except Exception as e:
            print(f"⚠️ 오류가 발생했습니다: {e}")
            break

    # 9. 모델 저장.
    classifier.save_assets()


if __name__ == "__main__":
    # GPU 메모리 사용 설정 (옵션)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    main()