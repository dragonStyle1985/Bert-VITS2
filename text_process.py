import os

out_file = f"filelists/田豫龙-红军不怕远征难.list"


def process():
    with open(out_file, 'w', encoding="Utf-8") as wf:
        ch_name = '田豫龙'   # 不需要和list的文件名一致
        ch_language = 'ZH'  # ZH, EN
        path = f"./raw/{ch_name}"
        files = os.listdir(path)
        sorted_files = sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x))))
        for f in sorted_files:
            if f.endswith(".lab"):
                with open(os.path.join(path, f), 'r', encoding="utf-8") as perFile:
                    line = perFile.readline()
                    result = f"./dataset/{ch_name}/{f.split('.')[0]}.wav|{ch_name}|{ch_language}|{line}"
                    wf.write(f"{result}\n")


if __name__ == "__main__":
    process()
