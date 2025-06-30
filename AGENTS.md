
这个项目是为了解决arc的ilp框架，非常简陋的框架，需要完善，参考的代码在目录: oldcodeforref/ 里面，
现在只是实现了arc数据的读取，入口程序文件计划是是mainpopperarc.py，希望你可以参考之前的oldcodeforref/ 代码来完善现在的代码

但是最近的改动代码入口都是：popper-arc/bkbias/objattr.py  现在的入口是 objattr.py  是现在最核心的文件。

现在代码的最新入口又是：mainpopperarc.py  了，已经可以循环多个任务了

ilp框架： <https://github.com/logic-and-learning-lab/Popper>
安装方法：pip install git+<https://github.com/logic-and-learning-lab/Popper@main>
   popper 请安装好

Library usage
You can import Popper and use it in your Python code like so:

from popper.util import Settings
from popper.loop import learn_solution

settings = Settings(kbpath='input_dir')
prog, score, stats = learn_solution(settings)
if prog != None:
    Settings.print_prog_score(prog, score)



SWI-Prolog 安装poppper要求的版本:方法：
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:swi-prolog/stable
sudo apt update
sudo apt install swi-prolog

 popper 请安装好

代码改动后请真实运行进行测试