import threading
import tkinter as tk
from tkinter import ttk, filedialog

from managers import NGCFManager
from managers.LightGCNManager import LightGCNManager
from managers.Manager import ManagerOption


MODELS = ["LightGCN", "NGCF"]
MODEL_MANAGER_MAP = {
    "LightGCN": LightGCNManager,
    "NGCF": NGCFManager,
}
DEVICES = ["cpu", "cuda:0"]
DATASETS = ["LastFM", "MovieLens"]


class GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("GNN Recommender GUI")
        self.master.minsize(960, 560)

        self.train_entries = {}
        self.infer_entries = {}
        self.option = ManagerOption()

        self.stop_flag = threading.Event()
        self.train_thread = None
        self.infer_thread = None
        self.train_stop_btn = None
        self.infer_stop_btn = None

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(padx=10, pady=10, fill="both", expand=True)

        self.train_tab = ttk.Frame(self.notebook)
        self.infer_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.train_tab, text="训练")
        self.notebook.add(self.infer_tab, text="推断")

        self.train_fields = [
            ("模型", "model_name", "combobox"),
            ("设备", "device", "combobox"),
            ("嵌入维度", "embed_dim", "entry"),
            ("层数", "num_layers", "entry"),
            ("TopN", "topN", "entry"),
            ("数据集", "dataset", "combobox"),
            ("数据路径", "data_dir", "dir"),
            ("批大小", "batch_size", "entry"),
            ("负样本数", "num_negatives", "entry"),
            ("正则化", "reg", "entry"),
            ("学习率", "lr", "entry"),
            ("Dropout", "dropout", "entry"),
            ("Epochs", "epochs", "entry"),
            ("评估频率", "eval_freq", "entry"),
            ("保存路径", "save_path", "dir"),
        ]

        self.infer_fields = [
            ("模型", "model_name", "combobox"),
            ("设备", "device", "combobox"),
            ("嵌入维度", "embed_dim", "entry"),
            ("层数", "num_layers", "entry"),
            ("TopN", "topN", "entry"),
            ("模型路径", "load_path", "dir"),
            ("用户id", "users", "entry"),
        ]

        self.train_console = self.build_tab(
            self.train_tab,
            self.train_fields,
            self.train_entries,
            btn_text="开始训练",
            btn_command=self.start_train,
            stop_attr="train_stop_btn"
        )
        self.load_option_to_gui(self.train_entries)

        self.infer_console = self.build_tab(
            self.infer_tab,
            self.infer_fields,
            self.infer_entries,
            btn_text="开始推断",
            btn_command=self.start_infer,
            stop_attr="infer_stop_btn"
        )
        self.load_option_to_gui(self.infer_entries)

    def build_field_inputs(self, parent, fields, entry_dict):
        for idx, (label, key, widget_type) in enumerate(fields):
            ttk.Label(parent, text=label).grid(row=idx, column=0, sticky="w", padx=5, pady=3)

            if widget_type == "entry":
                entry = tk.Entry(parent)
                entry.grid(row=idx, column=1, sticky="w", padx=5)
            elif widget_type == "combobox":
                values = MODELS if key == "model_name" else DEVICES if key == "device" else DATASETS if key == "dataset" else []
                entry = ttk.Combobox(parent, values=values, state="readonly")
                entry.grid(row=idx, column=1, sticky="w", padx=5)
                entry.current(0)
            elif widget_type == "dir":
                entry = tk.Entry(parent)
                entry.grid(row=idx, column=1, sticky="w", padx=5)
                btn = tk.Button(parent, text="选择", command=lambda e=entry: self.browse_directory(e))
                btn.grid(row=idx, column=2, sticky="w", padx=5)
            else:
                continue

            entry_dict[key] = entry

    def build_tab(self, tab_frame, fields, entries, btn_text, btn_command, stop_attr):
        for widget in tab_frame.winfo_children():
            widget.destroy()

        tab_frame.columnconfigure(0, weight=1)
        tab_frame.columnconfigure(1, weight=1)
        tab_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(tab_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        right_frame = ttk.Frame(tab_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.build_field_inputs(left_frame, fields, entries)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=len(fields), column=0, columnspan=3, pady=10, sticky="ew")
        btn_frame.columnconfigure((0, 1, 2), weight=1)

        action_btn = tk.Button(btn_frame, text=btn_text, command=btn_command)
        action_btn.grid(row=0, column=0, padx=5)

        stop_btn = tk.Button(btn_frame, text="停止", command=self.stop_process)
        stop_btn.grid(row=0, column=1, padx=5)
        stop_btn.grid_remove()  # 默认隐藏

        # 动态设置 self.train_stop_btn 或 self.infer_stop_btn
        setattr(self, stop_attr, stop_btn)

        clear_btn = tk.Button(btn_frame, text="清空输出")
        clear_btn.grid(row=0, column=2, padx=5)

        console = tk.Text(right_frame, height=30, width=80)
        console.pack(fill="both", expand=True)
        console.config(state="disabled")

        clear_btn.config(command=lambda c=console: self.clear_console(c))

        return console

    @staticmethod
    def browse_directory(entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    @staticmethod
    def write_console(console, text):
        def append():
            console.config(state=tk.NORMAL)
            console.insert(tk.END, text + "\n")
            console.see(tk.END)
            console.config(state=tk.DISABLED)

        console.after(0, append)

    @staticmethod
    def clear_console(console):
        console.config(state=tk.NORMAL)
        console.delete("1.0", tk.END)
        console.config(state=tk.DISABLED)

    def load_option_to_gui(self, entries):
        for key, widget in entries.items():
            if not hasattr(self.option, key):
                continue
            val = getattr(self.option, key)
            if val is None:
                continue
            if (key == "topN" or key == "users") and isinstance(val, (list, tuple)):
                val = " ".join(str(x) for x in val)
            else:
                val = str(val)

            if isinstance(widget, ttk.Combobox):
                if val in widget.cget("values"):
                    widget.set(val)
            elif isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
                widget.insert(0, val)

    def collect_args(self, entries, is_train=True):
        setattr(self.option, 'is_train', is_train)
        setattr(self.option, 'listen_events', True)
        setattr(self.option, 'stop_flag', self.stop_flag)
        for key, widget in entries.items():
            val = widget.get()
            if key in ["embed_dim", "num_layers", "num_negatives", "batch_size", "epochs", "eval_freq"]:
                val = int(val) if val.isdigit() else 0
            elif key in ["lr", "reg", "dropout"]:
                try:
                    val = float(val)
                except ValueError:
                    val = 0.0
            elif key == "topN" or key == "users":
                try:
                    val = list(map(int, val.split()))
                except ValueError:
                    val = [5]
            setattr(self.option, key, val)

    def stop_process(self):
        self.stop_flag.set()
        self.write_console(self.train_console, "已请求停止操作。")

    def start_train(self):
        self.collect_args(self.train_entries, is_train=True)
        self.write_console(self.train_console, "开始训练...")
        self.stop_flag.clear()
        self.train_stop_btn.grid()

        def train_thread():
            model_name = self.option.model_name
            manager_cls = MODEL_MANAGER_MAP.get(model_name)
            if manager_cls:
                manager = manager_cls(self.option, self.write_console, self.train_console)
                manager.train()
                self.write_console(self.train_console, "训练完成。")
            else:
                self.write_console(self.train_console, f"未知模型: {model_name}")
            self.train_stop_btn.grid_remove()

        self.train_thread = threading.Thread(target=train_thread, daemon=True)
        self.train_thread.start()

    def start_infer(self):
        self.collect_args(self.infer_entries, is_train=False)
        self.write_console(self.infer_console, "开始推断...")
        self.stop_flag.clear()
        self.infer_stop_btn.grid()

        def infer_thread():
            model_name = self.option.model_name
            manager_cls = MODEL_MANAGER_MAP.get(model_name)
            if manager_cls:
                manager = manager_cls(self.option, self.write_console, self.infer_console)
                manager.predict()
                self.write_console(self.infer_console, "推断完成。")
            else:
                self.write_console(self.infer_console, f"未知模型: {model_name}")
            self.infer_stop_btn.grid_remove()

        self.infer_thread = threading.Thread(target=infer_thread, daemon=True)
        self.infer_thread.start()
