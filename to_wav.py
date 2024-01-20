from pydub import AudioSegment

if __name__ == "__main__":

    # 定义输入和输出文件路径
    input_file = "E:/workspace/python/Bert-VITS2/data/zxn/你们那下那么大雪呀.m4a"
    output_file = "E:/workspace/python/Bert-VITS2/data/zxn/zxn_1.wav"

    # 加载.m4a文件并转换为AudioSegment对象
    audio = AudioSegment.from_file(input_file)

    # 导出为.wav文件
    audio.export(output_file, format="wav")
