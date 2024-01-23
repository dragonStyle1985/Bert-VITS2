import librosa
import opencc
import os
import whisper

import numpy as np

from pathlib import Path
from scipy.io import wavfile


a = "田豫龙-红军不怕远征难"  # 请在这里修改说话人的名字，将音频放在“data/人名”下


def split_long_audio(model, filepaths, save_dir="data_dir", out_sr=44100):
    files = os.listdir(filepaths)
    filepaths = [os.path.join(filepaths, i) for i in files]

    wav_index = 0
    for file_idx, filepath in enumerate(filepaths):

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        print(f"Transcribing file {file_idx}: '{filepath}' to segments...")
        result = model.transcribe(filepath, word_timestamps=True, task="transcribe", beam_size=5, best_of=5)
        segments = result['segments']

        wav, sr = librosa.load(filepath, sr=None, offset=0, duration=None, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=out_sr)
        wav2 /= max(wav2.max(), -wav2.min())

        for i, seg in enumerate(segments):
            start_time = seg['start']
            end_time = seg['end']
            wav_seg = wav2[int(start_time * out_sr):int(end_time * out_sr)]
            wav_seg_name = f"{a}_{wav_index}.wav"  # 修改名字
            wav_index += 1
            out_fpath = save_path / wav_seg_name
            wavfile.write(out_fpath, rate=out_sr, data=(wav_seg * np.iinfo(np.int16).max).astype(np.int16))


def transcribe_one(audio_path, converter):  # 使用whisper语音识别
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)
    simplified_text = converter.convert(result.text)
    # print the recognized text
    print(simplified_text)
    return simplified_text


if __name__ == '__main__':
    whisper_size = "medium"
    model = whisper.load_model(whisper_size)
    audio_path = f"./raw/{a}"
    if os.path.exists(audio_path):
        for filename in os.listdir(audio_path):  # 删除raw目录下原来的音频和文本
            file_path = os.path.join(audio_path, filename)
            os.remove(file_path)
    split_long_audio(model, f"data/{a}", f"./raw/{a}")
    files = os.listdir(audio_path)
    file_list_sorted = sorted(files, key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
    filepaths = [os.path.join(audio_path, i) for i in file_list_sorted]

    # Convert the recognized text from traditional Chinese to simplified Chinese
    converter = opencc.OpenCC('t2s')  # 使用简繁体转换配置文件
    for file_idx, filepath in enumerate(filepaths):  # 循环使用whisper遍历每一个音频,写入.lab
        text = transcribe_one(filepath, converter)
        with open(f"./raw/{a}/{a}_{file_idx}.lab", 'w', encoding='utf-8') as f:
            f.write(text)

