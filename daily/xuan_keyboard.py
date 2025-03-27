# rainbow_yu cron.task_schedule 🐋✨
# 定时任务

import time
import threading
from datetime import datetime
from collections import defaultdict
import pyautogui
import subprocess
import os
import threading
import jieba
from pypinyin import lazy_pinyin

class TimedTask:
    def __init__(
            self,
            format = '%H:%M:%S'
    ):
        """初始化定时任务管理器"""
        self.task_schedule = {}  # 存储按时间调度的任务
        self.running_tasks = {}  # 存储正在运行的任务
        self.dependent_tasks = defaultdict(list)  # 存储依赖任务（任务完成后启动）
        self.format = format

    def start_thread(self, task_func, task_name, *args):
        """启动任务线程"""
        stop_event = threading.Event()  # 创建停止事件
        thread = threading.Thread(target=self.run_task, args=(task_func, stop_event, task_name, *args))
        thread.daemon = True  # 设置为守护线程
        self.running_tasks[task_name] = stop_event
        thread.start()

    def run_task(self, task_func, stop_event, task_name, *args):
        """执行任务线程"""
        try:
            print(f"任务 {task_name} 在 {datetime.now()} 开始")
            task_func(*args)  # 执行任务
            time.sleep(5)  # 模拟任务执行时间
        except Exception as e:
            print(f"任务 {task_name} 发生错误: {e}")
        finally:
            print(f"任务 {task_name} 完成，在 {datetime.now()} 停止")
            stop_event.set()  # 任务结束
            self.running_tasks.pop(task_name, None)  # 移除运行中的任务
            self.execute_dependent_tasks(task_name)  # 仅在任务结束后执行依赖任务

    def execute_dependent_tasks(self, completed_task):
        """执行依赖于指定任务的任务"""
        if completed_task in self.dependent_tasks:
            # 复制依赖任务列表，并重新添加回去
            tasks_to_execute = list(self.dependent_tasks[completed_task])
            for task_func, task_name, args in tasks_to_execute:
                print(f"任务 {completed_task} 完成，启动依赖任务 {task_name}")
                self.start_thread(task_func, task_name, *args)

            # 重新注册依赖任务（保持依赖关系不会被清除）
            self.dependent_tasks[completed_task] = tasks_to_execute

    def schedule(self):
        """根据当前时间调度任务"""
        current_time = datetime.now()
        to_remove = []

        for task_name, task_info in self.task_schedule.items():
            start_time = task_info["start_time"]
            end_time = task_info["end_time"]
            task_func = task_info["task_func"]
            task_args = task_info["task_args"]

            today_start_time = current_time.replace(hour=start_time.hour, minute=start_time.minute, second=0,
                                                    microsecond=0)
            # If end_time is None, use it as a large value to let the task run until it's stopped.
            today_end_time = current_time.replace(hour=23, minute=59, second=59, microsecond=0) if end_time is None else current_time.replace(hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0)

            if today_start_time <= current_time <= today_end_time:
                if task_name not in self.running_tasks:
                    print(f"任务 {task_name} 在 {current_time} 启动")
                    self.start_thread(task_func, task_name, *task_args)
                # If end_time is None, remove it from the schedule after the task is done.
                if end_time is None:
                    to_remove.append(task_name)

            elif current_time > today_end_time:
                if task_name in self.running_tasks:
                    print(f"任务 {task_name} 在 {current_time} 停止")
                    self.running_tasks[task_name].set()  # 停止任务
                    to_remove.append(task_name)

        for task_name in to_remove:
            self.running_tasks.pop(task_name, None)
            # Remove tasks with None end_time after they've completed
            if task_name in self.task_schedule and self.task_schedule[task_name]["end_time"] is None:
                del self.task_schedule[task_name]

        time.sleep(5)
        self.schedule()

    def add_task(self, task_name, start_time_str, end_time_str, task_func, *args):
        """添加定时任务"""
        start_time = datetime.strptime(start_time_str, self.format)
        # If end_time_str is None, set end_time to None
        end_time = None if end_time_str is None else datetime.strptime(end_time_str, self.format)
        self.task_schedule[task_name] = {
            'start_time': start_time,
            'end_time': end_time,
            'task_func': task_func,
            'task_args': args
        }
        print(f"添加任务: {task_name}，时间: {start_time.strftime(self.format)} ~ {('不结束' if end_time is None else end_time.strftime(self.format))}")

    def add_dependent_task(self, after_task, task_name, task_func, *args):
        """添加依赖任务（某个任务完成后才执行）"""
        self.dependent_tasks[after_task].append((task_func, task_name, args))
        print(f"添加依赖任务: {task_name}，将在 {after_task} 任务完成后执行")



# rainbow_yu cron.time_transform 🐋✨
# 时间戳转换

from datetime import datetime, timedelta


class TimeTransform:
    def __init__(
            self,
            input_time,
            end_time=None,
            format='%H:%M:%S'
    ):
        """
        时间转换类
        :param input_time: 输入时间/开始时间
        :param end_time: 结束时间
        """
        self.current_time = datetime.now()
        self.format = format

        # 将字符串时间转换为 datetime 对象
        if isinstance(input_time, str):
            self.input_time = datetime.strptime(input_time, self.format)
        elif isinstance(input_time, datetime):
            self.input_time = input_time
        else:
            raise TypeError

        if end_time and isinstance(end_time, str):
            self.end_time = datetime.strptime(end_time, self.format)
        elif end_time and isinstance(end_time, datetime):
            self.end_time = end_time
        else:
            self.end_time = None  # 如果没有传入 end_time，设置为 None

        # 设置转换后的时间
        self.transformed_time = self.current_time.replace(
            hour=self.input_time.hour, minute=self.input_time.minute, second=input_time.second, microsecond=0
        )

    def get_current_time(self):
        """
        获取当前时间
        :return: 当前时间
        """
        self.current_time = datetime.now()
        return self.current_time

    def to_string(self):
        return self.transformed_time.strftime("%H:%M:%S")

    def __repr__(self):
        return str(self.transformed_time)

    def time_compare(self, operator):
        """
        目标时间和当前时间的比较
        :param operator: 操作符
        :return: bool - 比较结果
        """
        valid_operators = ['>', '<', '=', '>=', '<=']
        if operator not in valid_operators:
            raise ValueError(f"Invalid operator. Please use one of: {valid_operators}")

        # Perform the comparison based on the operator
        if operator == '>':
            return self.transformed_time > self.current_time
        elif operator == '<':
            return self.transformed_time < self.current_time
        elif operator == '=':
            return self.transformed_time == self.current_time
        elif operator == '>=':
            return self.transformed_time >= self.current_time
        elif operator == '<=':
            return self.transformed_time <= self.current_time

    def cal_lasting_time(self, buffer_time=0):
        """
        计算持续时间并返回 lasting_time
        :return: lasting_time 格式为 "HH:MM:SS"
        """
        buffer_time = timedelta(seconds=buffer_time)
        if self.end_time and self.input_time:  # 确保 input_time 和 end_time 都存在
            if isinstance(self.end_time, datetime) and isinstance(self.input_time, datetime):
                # 计算持续时间
                lasting_time_delta = self.end_time - self.input_time - buffer_time

                # 计算小时、分钟和秒
                hours, remainder = divmod(lasting_time_delta.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)

                # 如果持续时间超过24小时，处理天数
                days = lasting_time_delta.days
                if days > 0:
                    hours += days * 24  # 加上天数转换为小时

                # 格式化时间为 "HH:MM:SS"
                self.lasting_time = f"{hours:02}:{minutes:02}:{seconds:02}"

                return self.lasting_time
            else:
                raise ValueError("Both input_time and end_time must be datetime objects.")
        else:
            raise ValueError("Both input_time and end_time are required to calculate lasting time.")


def open_markdown():
    """创建并打开Markdown文件"""
    file_name = "my_notes.md"

    # 如果文件存在，先删除
    if os.path.exists(file_name):
        os.remove(file_name)

    # 创建新文件
    with open(file_name, "w") as file:
        file.write("# 我的笔记\n")  # 创建一个带有标题的空的Markdown文件

    # 打开Markdown文件
    subprocess.run(["start", file_name], check=True, shell=True)  # 依赖默认Markdown编辑器打开


def type_text(chinese_text=None, english_text=None):
    """等待记事本打开并输入文本（中文转换为拼音，英文直接输入）"""
    pyautogui.press('esc')  # 关闭输入法，如果是中文输入法可能需要这一步
    time.sleep(1)

    # 输入英文文本
    if english_text:
        pyautogui.press('shift')
        time.sleep(0.1)
        lines = english_text.split("\n")
        for line in lines:
            # 直接模拟键盘输入英文及符号
            pyautogui.write(line, interval=0.01)
            pyautogui.press('enter')  # 换行

    # 输入中文文本（转换为拼音）
    if chinese_text:
        lines = chinese_text.split("\n")
        for line in lines:
            # 使用 jieba 进行分词
            words = jieba.lcut(line)  # 分词
            print("分词结果：", words)  # 调试用，可删除

            # 将整个词转换为拼音
            pinyin_text = " ".join(["".join(lazy_pinyin(word)) for word in words])

            # 模拟键盘输入拼音
            pyautogui.write(pinyin_text, interval=0.2)

            # 模拟按下回车键（换行）
            pyautogui.press('space')
            pyautogui.press('enter')



if __name__ == "__main__":
    time_manager = TimedTask()

    chinese_text = """下午好
吴依萱小姐姐
俞天翼来看看你啦
今天过的怎么样呀
希望你每天都开开心心的
没有烦恼
往后运气超好
心想事成
能去想去的学校
见到我的时候也是快快乐乐的
等一下在执行定时任务哦"""
    english_text = r"""
#         __                
#        /\ \__             
#  __  __\ \ ,_\  __  __    
# /\ \/\ \\ \ \/ /\ \/\ \   
# \ \ \_\ \\ \ \_\ \ \_\ \  
#  \/`____ \\ \__\\/`____ \ 
#   `/___/> \\/__/ `/___/> \
#      /\___/         /\___/
#      \/__/          \/__/ 
# 
#  ___                             
# /\_ \                            
# \//\ \     ___   __  __     __   
#   \ \ \   / __`\/\ \/\ \  /'__`\ 
#    \_\ \_/\ \L\ \ \ \_/ |/\  __/ 
#    /\____\ \____/\ \___/ \ \____\
#    \/____/\/___/  \/__/   \/____/
#                                  
#  __  __  __  __  __   __  _  
# /\ \/\ \/\ \/\ \/\ \ /\ \/'\ 
# \ \ \_/ \_/ \ \ \_\ \\/>  </ 
#  \ \___x___/'\/`____ \/\_/\_\
#   \/__//__/   `/___/> \//\/_/
#                  /\___/      
#                  \/__/       
# 
#    ___                                            
#  /'___\                                           
# /\ \__/  ___   _ __    __   __  __     __   _ __  
# \ \ ,__\/ __`\/\`'__\/'__`\/\ \/\ \  /'__`\/\`'__\
#  \ \ \_/\ \L\ \ \ \//\  __/\ \ \_/ |/\  __/\ \ \/ 
#   \ \_\\ \____/\ \_\\ \____\\ \___/ \ \____\\ \_\ 
#    \/_/ \/___/  \/_/ \/____/ \/__/   \/____/ \/_/ 
"""

    t = TimeTransform(datetime.now())
    t_deta = TimeTransform(datetime.now()+timedelta(seconds=10))

    time_manager.add_task("run_note",t.to_string(),None,open_markdown)
    time_manager.add_task("type_chinese",t_deta.to_string(),None,type_text,chinese_text,None)
    time_manager.add_dependent_task("type_chinese","type_english",type_text,None,english_text)

    time_manager.schedule()

