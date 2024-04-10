import torch
import sys
import json
from playsound import playsound
from pathlib import Path
from datetime import datetime
from diffusers import DiffusionPipeline
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QTextEdit, QLabel, QFormLayout, QGroupBox, \
    QSpinBox, QFileDialog, QLineEdit, QComboBox, QCheckBox, QFrame

ICON_PATH = "resource/diffusion_runner.ico"
COMPLETE_SOUND = "resource/complete_seq.wav"
PIC_MADE_SOUND = "resource/made_a_pic.wav"
LOADED_SOUND = "resource/model_loaded.wav"

DEFAULT_PROMPT = "general, masterpiece, very aesthetic"
DEFAULT_NEGATIVE_PROMPT = "lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, " \
                          "low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, " \
                          "signature, extra digits,nsfw, sensitive, artistic error, username, scan, [abstract] "

RESOLUTIONS_STR = ["1024 x 1024 1:1 Square", "1152 x 896 9:7", "896 x 1152 7:9", "1216 x 832 19:13", "832 x 1216 13:19",
                   "1344 x 768 7:4 Horizontal", "768 x 1344 4:7 Vertical", "1536 x 640 12:5 Horizontal",
                   "640 x 1536 5:12 Vertical", "1920 x 1080 16:9", "512 x 512"]
RESOLUTIONS = [[1024, 1024], [1152, 896], [896, 1152], [1216, 832], [832, 1216], [1344, 768], [768, 1344], [1536, 640],
               [640, 1536], [1920, 1080], [512, 512]]

# TODO: Solve need for hacky global variables
_pipe = None
_start_button = None

_config_dict: dict = {"default_model": ".", "default_prompt": DEFAULT_PROMPT,
                      "default_negative_prompt": DEFAULT_NEGATIVE_PROMPT}


def get_main_dir() -> str:
    main_dir = str(Path(__file__).parent.absolute())
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        main_dir = sys._MEIPASS
    return main_dir


def open_directory_dialog(lineEdit: QLineEdit):
    directory = QFileDialog.getExistingDirectory()
    lineEdit.setText('{}'.format(directory))


def open_file_dialog():
    dialog = QFileDialog()
    dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
    dialog.setNameFilter("Text files (*.txt)")
    if dialog.exec_():
        return dialog.selectedFiles()


def edit_config(key: str, value):
    _config_dict[key] = value
    save_config()


def save_config():
    global _config_dict
    file_name = "diff_runner_config.json"
    path = Path(__file__).parent.absolute() / file_name
    with open(path, 'w') as json_file:
        json.dump(_config_dict, json_file)


def load_config():
    global _config_dict
    file_name = "diff_runner_config.json"
    path = Path(__file__).parent.absolute() / file_name
    if path.is_file():
        with open(path, 'r') as config_file:
            _config_dict = json.load(config_file)
    else:
        save_config()


def save_prompt_to_txt(prompt: str):
    file_name = QFileDialog().getSaveFileName(filter="Text files (*.txt)")[0]
    with open(file_name, 'w') as f:
        f.write(prompt)


def load_prompt_from_txt(textbox: QTextEdit):
    file_name = open_file_dialog()[0]
    with open(file_name, 'r') as f:
        prompt = f.read()
        textbox.setText(prompt)


def load_model(model: str):
    global _pipe
    global _start_button
    _pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    _pipe.to('cuda')
    playsound(get_main_dir() + "/" + LOADED_SOUND)
    _start_button.setDisabled(False)


def generate(prompt: str, negative_prompt: str, resolution: list[2], inf_steps: int, iteration: int, directory: str,
             pipe):
    for idx in range(0, iteration):
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            width=resolution[0],
            height=resolution[1],
            guidance_scale=7,
            num_inference_steps=inf_steps
        ).images[0]

        time_stamp = datetime.timestamp(datetime.now())
        file_name = f"test_{time_stamp}.png"

        path = Path(directory) / file_name
        image.save(path)
        playsound(get_main_dir() + "/" + PIC_MADE_SOUND)
        print("done")
    playsound(get_main_dir() + "/" + COMPLETE_SOUND)


class model_group(QGroupBox):
    def __init__(self):
        super(model_group, self).__init__()

        global _config_dict

        self.setTitle("Model")
        model_form = QFormLayout()

        model_directory_hbox = QHBoxLayout()
        model_directory_line = QLineEdit()
        model_directory_line.setText(_config_dict["default_model"])
        model_browse_button = QPushButton("Browse")
        model_browse_button.clicked.connect(lambda: open_directory_dialog(model_directory_line))
        model_directory_hbox.addWidget(QLabel("Model Directory"))
        model_directory_hbox.addWidget(model_directory_line)
        model_directory_hbox.addWidget(model_browse_button)

        model_load_button = QPushButton("Load Model")
        model_load_button.clicked.connect(lambda: load_model(model_directory_line.text()))

        model_form.addRow(model_directory_hbox)
        model_form.addRow(model_load_button)
        self.setLayout(model_form)


class prompt_group(QGroupBox):
    def __init__(self, main_title: str = "Prompts", prompt_title: str = "Prompt",
                 default_prompt_title: str = "Default Prompt", name: str = "default_prompt",
                 has_char_mode: bool = False):
        super(prompt_group, self).__init__()
        self.setTitle(main_title)
        prompt_form = QFormLayout()

        self.character_mode_checkbox = QCheckBox("Character Mode:")
        self.character_mode_checkbox.clicked.connect(lambda: self.toggle_character_mode_prompt_box())

        self.prompt_textbox = QTextEdit()

        default_prompt_hbox = QHBoxLayout()
        self.default_prompt_textbox = QTextEdit(_config_dict[name])
        save_default_prompt = QPushButton("Save default")
        save_default_prompt.clicked.connect(lambda: edit_config(name, self.default_prompt_textbox.toPlainText()))
        default_prompt_hbox.addWidget(self.default_prompt_textbox)
        default_prompt_hbox.addWidget(save_default_prompt)

        if has_char_mode:
            self.character_prompt_frame = QFrame()
            character_prompt_form = QFormLayout()
            character_prompt_hbox = QHBoxLayout()
            self.character_prompt_textbox = QTextEdit()
            save_character_prompt_button = QPushButton("Save")
            save_character_prompt_button.clicked.connect(
                lambda: save_prompt_to_txt(self.character_prompt_textbox.toPlainText()))
            load_character_prompt_button = QPushButton("Load")
            load_character_prompt_button.clicked.connect(lambda: load_prompt_from_txt(self.character_prompt_textbox))
            character_prompt_hbox.addWidget(self.character_prompt_textbox)
            character_prompt_hbox.addWidget(save_character_prompt_button)
            character_prompt_hbox.addWidget(load_character_prompt_button)
            character_prompt_form.addRow(QLabel("Character Form"))
            character_prompt_form.addRow(character_prompt_hbox)
            self.character_prompt_frame.setLayout(character_prompt_form)
            self.character_prompt_frame.hide()

        if has_char_mode:
            prompt_form.addRow(self.character_mode_checkbox)
        prompt_form.addRow(QLabel(prompt_title))
        prompt_form.addRow(self.prompt_textbox)
        prompt_form.addRow(QLabel(default_prompt_title))
        prompt_form.addRow(default_prompt_hbox)
        if has_char_mode:
            prompt_form.addRow(self.character_prompt_frame)
        self.setLayout(prompt_form)

    def toggle_character_mode_prompt_box(self):
        if self.character_mode_checkbox.isChecked():
            self.character_prompt_frame.show()
        else:
            self.character_prompt_frame.hide()

    def get_prompt(self) -> str:
        char_prompt = ""
        if self.character_mode_checkbox.isChecked():
            char_prompt = self.character_prompt_textbox.toPlainText() + ","
        prompt: str = char_prompt + self.prompt_textbox.toPlainText() + "," + self.default_prompt_textbox.toPlainText()
        return prompt


class generation_group(QGroupBox):
    def __init__(self, prompt_box: prompt_group, negative_prompt_box: prompt_group):
        super(generation_group, self).__init__()
        global _pipe

        self.setTitle("Generation")
        generation_form = QFormLayout()

        resolution_hbox = QHBoxLayout()
        resolution_combo = QComboBox()
        resolution_combo.addItems(RESOLUTIONS_STR)
        resolution_hbox.addWidget(QLabel("Resolution:"))
        resolution_hbox.addWidget(resolution_combo)

        inference_steps_hbox = QHBoxLayout()
        inference_steps_hbox.addWidget(QLabel("Number of inference steps:"))
        inference_steps_spinbox = QSpinBox()
        inference_steps_spinbox.setValue(28)
        inference_steps_spinbox.setMinimum(1)
        inference_steps_spinbox.setMaximum(200)
        inference_steps_hbox.addWidget(inference_steps_spinbox)

        number_gen_hbox = QHBoxLayout()
        number_gen_hbox.addWidget(QLabel("Number of image to generate:"))
        number_gen_spinbox = QSpinBox()
        number_gen_spinbox.setValue(1)
        number_gen_spinbox.setMinimum(1)
        number_gen_spinbox.setMaximum(20)
        number_gen_hbox.addWidget(number_gen_spinbox)

        directory_hbox = QHBoxLayout()
        directory_line = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(lambda: open_directory_dialog(directory_line))
        directory_hbox.addWidget(QLabel("Image directory"))
        directory_hbox.addWidget(directory_line)
        directory_hbox.addWidget(browse_button)

        global _start_button
        _start_button = QPushButton("Start")
        _start_button.clicked.connect(lambda: generate(prompt=prompt_box.get_prompt(),
                                                       negative_prompt=negative_prompt_box.get_prompt(),
                                                       resolution=RESOLUTIONS[resolution_combo.currentIndex()],
                                                       inf_steps=inference_steps_spinbox.value(),
                                                       iteration=number_gen_spinbox.value(),
                                                       directory=directory_line.text(),
                                                       pipe=_pipe))
        _start_button.setDisabled(True)

        generation_form.addRow(resolution_hbox)
        generation_form.addRow(inference_steps_hbox)
        generation_form.addRow(number_gen_hbox)
        generation_form.addRow(directory_hbox)
        generation_form.addRow(_start_button)
        self.setLayout(generation_form)


if __name__ == '__main__':
    app = QApplication([])
    load_config()
    app.setWindowIcon(QIcon(get_main_dir() + "/" + ICON_PATH))
    model_group_box = model_group()
    positive_prompt_box = prompt_group(has_char_mode=True)
    negative_prompt_box = prompt_group(main_title="Negative Prompts", prompt_title="Negative Prompt",
                                       default_prompt_title="Default Negative Prompt",
                                       name="default_negative_prompt")

    generation_group_box = generation_group(prompt_box=positive_prompt_box, negative_prompt_box=negative_prompt_box)

    # Global layout
    layout = QFormLayout()
    layout.addRow(model_group_box)
    layout.addRow(positive_prompt_box)
    layout.addRow(negative_prompt_box)
    layout.addRow(generation_group_box)

    # Main window
    window = QWidget()
    window.setWindowTitle("Diffusion Runner")
    window.setLayout(layout)
    # TODO: Improve size setting to be less rigid
    window.setMinimumSize(600, 800)
    window.show()
    sys.exit(app.exec())
