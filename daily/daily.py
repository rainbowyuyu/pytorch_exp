import random

def random_chinese_character():
    # 常见汉字基本范围：\u4E00-\u9FA5
    return chr(random.randint(0x4E00, 0x9FA5))

if __name__ == '__main__':
    for i in range(10):
        for j in range(30):
            print(random_chinese_character(),end=" ")
        print()
