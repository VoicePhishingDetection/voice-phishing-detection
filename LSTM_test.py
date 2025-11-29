import numpy as np
import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras 임포트
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 60)
print("스팸 메일 탐지 LSTM 프로젝트 (실제 데이터셋 사용)")
print("=" * 60)

# ============================================================
# 1. 실제 데이터셋 로드 및 전처리
# ============================================================

DATASET_FILE = 'spam.csv'

print(f"\n[1단계] 데이터셋 로드 및 정리 ({DATASET_FILE} 사용 중)...")

try:
    # CSV 파일 로드: 인코딩 문제 방지를 위해 'latin-1' 인코딩 사용
    # 필요한 두 개의 열만 읽고, 열 이름 변경 (v1 -> label, v2 -> text)
    df = pd.read_csv(DATASET_FILE, encoding='latin-1', usecols=['v1', 'v2'])
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})

    # 'label'을 숫자(0:ham, 1:spam)로 인코딩
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # 데이터셋의 일부 열이 필요 없으므로 제거 (컬럼 3개 이상일 경우)
    if df.shape[1] > 2:
        df = df.iloc[:, :2]

    emails = df['text'].tolist()
    labels = df['label'].tolist()

    print(f"✓ 데이터 로드 완료. 총 {len(emails)}개 문장")
    print(f"  - 스팸: {df['label'].sum()}개")
    print(f"  - 정상: {len(df) - df['label'].sum()}개")

except FileNotFoundError:
    print(f"⚠️ 오류: 파일 '{DATASET_FILE}'을 찾을 수 없습니다. 파일 이름을 확인해 주세요.")
    exit()
except Exception as e:
    print(f"⚠️ 오류 발생: 데이터 로드 및 정리 실패 - {e}")
    exit()

# ============================================================
# 2. 텍스트 전처리
# ============================================================

print("\n[2단계] 텍스트 전처리 중...")


def preprocess_text(text):
    """텍스트 전처리 함수 (소문자 변환, 특수문자 제거, 공백 정리)"""
    text = text.lower()
    # 문장부호 및 특수문자 제거 (알파벳, 숫자, 공백만 유지)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로
    return text.strip()


# 전처리 적용
processed_emails = [preprocess_text(email) for email in emails]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    processed_emails, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"✓ 전처리 완료")
print(f"  - 학습 데이터: {len(X_train)}개")
print(f"  - 테스트 데이터: {len(X_test)}개")

# ============================================================
# 3. 토크나이징 및 시퀀스 변환
# ============================================================

print("\n[3단계] 토크나이징 및 시퀀스 변환 중...")

# 토크나이저 설정
MAX_WORDS = 10000  # 어휘 크기 (동일 유지)
MAX_LEN = 100  # 최대 시퀀스 길이 (동일 유지)

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# 시퀀스 변환 및 패딩
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post', truncating='post')

# numpy 배열로 변환
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"✓ 토크나이징 완료")
print(f"  - 어휘 크기: {len(tokenizer.word_index)}개")
print(f"  - 시퀀스 길이: {MAX_LEN}")
print(f"  - 학습 데이터 shape: {X_train_pad.shape}")

# ============================================================
# 4. LSTM 모델 구축
# ============================================================

print("\n[4단계] LSTM 모델 구축 중...")

# 모델 구조는 동일하게 유지
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✓ 모델 구축 완료")
model.summary()
#


# ============================================================
# 5. 모델 학습
# ============================================================

print("\n[5단계] 모델 학습 시작...")

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("\n✓ 모델 학습 완료")

# ============================================================
# 6. 모델 평가
# ============================================================

print("\n[6단계] 모델 평가...")

y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

cm = confusion_matrix(y_test, y_pred)
print("\n혼동 행렬:")
print(cm)

test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"\n테스트 정확도: {test_acc:.4f}")


# ============================================================
# 7. 실시간 예측 함수 (임계값 0.6 적용)
# ============================================================

def predict_email(text):
    """이메일 텍스트를 입력받아 스팸 여부 예측 (임계값 0.6 적용)"""
    processed = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]

    # **엄격한 스팸 분류를 위해 임계값 0.6 유지**
    THRESHOLD = 0.6

    label = "스팸 (Spam)" if prediction > THRESHOLD else "정상 (Ham)"
    # 신뢰도는 임계값 0.6을 기준으로 더 높은 확률값을 사용합니다.
    confidence = prediction if prediction > THRESHOLD else 1 - prediction

    return label, confidence


# ============================================================
# 8. 사용자 입력 실시간 예측
# ============================================================

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

        label, confidence = predict_email(user_input)

        # 예측 결과 출력
        print("-" * 30)
        print(f"**이메일:** {user_input}")
        print(f"**예측 결과:** {label}")
        print(f"**신뢰도:** {confidence:.2%}")
        print("-" * 30)

    except Exception as e:
        print(f"⚠️ 오류가 발생했습니다: {e}")
        break

print("\n" + "=" * 60)
print("프로젝트 완료!")
print("=" * 60)

# 모델 및 토크나이저 저장 (추후 재사용을 위해)
model.save('real_spam_detector_lstm.keras')
print("\n✓ 모델 저장: real_spam_detector_lstm.keras")

import pickle

with open('real_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✓ 토크나이저 저장: real_tokenizer.pkl")