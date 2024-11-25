import os
import sys
import subprocess

# 环境变量的名称和值
env_var_name = "MMDLP_TOOLS_PATH"
env_var_value = os.path.abspath(os.path.join(os.path.dirname(__file__)))

bash_path = os.path.join(os.path.expanduser('~'), ".bashrc")

# 要添加到.bashrc的行
line_to_add = f'export {env_var_name}="{env_var_value}"'


# 检查环境变量是否已经存在于.bashrc中
def is_env_var_in_bashrc():
    with open(bash_path, 'r') as file:
        for line in file:
            if env_var_name in line:
                return True
    return False


# 将环境变量添加到.bashrc
def add_env_var_to_bashrc():
    if not is_env_var_in_bashrc():
        with open(bash_path, 'a') as file:
            file.write(f'\n{line_to_add}\n')
    else:
        print(f"环境变量 {env_var_name} 已经存在于 .bashrc 中。")


# 重新加载.bashrc
def reload_bashrc():
    subprocess.run(['source', '~/.bashrc'], shell=True)


if __name__ == '__main__':
    add_env_var_to_bashrc()
    print(f"setting {env_var_name}={env_var_value})")