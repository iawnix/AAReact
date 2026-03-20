from typing import Union
import subprocess


def CMD_RUN(cmd: str, printsign: bool = False) -> Union[str, bool]:
    """
    这个函数执行grep的时候会有异常, 即使没有grep到目标也会返回out1, 而不是False
    """

    ret1 = subprocess.Popen(
        cmd,
        bufsize=-1,
        shell=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ret1_ = ret1.communicate(input=None)
    out1, error1 = ret1_[0], ret1_[1]
    code = ret1.returncode
    if error1 != "":
        if not code:                    # 解决UBUNTU上浮点数的Note导致程序跳出
            if printsign:
                print("Sucessful: {},But: {}".format(cmd, error1))
            return out1
        else:
            if printsign:
                print("Error: {}".format(error1))
            return False
    else:
        if printsign:
            print("Sucessful: {}".format(cmd))
        return out1

