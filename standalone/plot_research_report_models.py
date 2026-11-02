#!/usr/bin/env python3
"""Draw explicitly labelled analytical schematics, not measured results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/project_reports/2026阶段研究_系统调研详版/figures"
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(),
                         "axes.unicode_minus": False, "font.size": 11})
    x,y=np.meshgrid(np.linspace(-8,8,151),np.linspace(-8,8,151))
    losses=[np.zeros_like(x),(.2*x+.1*y)**2,(.2*x+.1*y)**2+(-.1*x+.2*y)**2]
    names=["平坦：两个水平方向均缺信息","单坡：等高线方向仍弱","方向丰富：局部平移更可辨"]
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharex=True,sharey=True)
    for ax,loss,title in zip(axes,losses,names):
        ax.imshow(loss,origin="lower",extent=(-8,8,-8,8),cmap="YlGnBu",vmin=0,vmax=5)
        if np.max(loss)>0:
            ax.contour(x,y,loss,levels=[.1,.5,1,2],colors="white",linewidths=.8)
        ax.set_title(title,fontsize=11)
        ax.set_xlabel("水平位置扰动 x / m")
        ax.scatter([0],[0],color="#cf4e29",s=30)
    axes[0].set_ylabel("水平位置扰动 y / m")
    fig.suptitle("地形残差局部信息示意（解析构造，不是实验数据）",fontsize=15)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(OUT/"地形可观性解析示意.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for n,color in zip(range(4),["#1565c0","#e67322","#259b70","#965bc0"]):
        left=n*.3
        axes[0].plot([left,left+1.2],[n,n],lw=9,alpha=.6,color=color,solid_capstyle="butt")
        axes[0].scatter(left+np.arange(8)*.15,np.full(8,n),s=17,color=color)
    axes[0].set_yticks(range(4),["ping 1","ping 2","ping 3","ping 4"])
    axes[0].set_xlabel("沿航向名义支持 / m")
    axes[0].set_title("同步8行与连续4 ping（5 Hz，1.5 m/s）")
    axes[0].grid(axis="x",alpha=.2)
    r=np.linspace(1,250,400)
    resolution=np.where(r<=60,.15,np.sin(np.deg2rad(.14))*r)
    axes[1].plot(r,resolution,color="#1565c0",lw=2,label="标准模式名义分辨率")
    axes[1].axhline(.15,color="#e67322",ls="--",label="成像行格距 0.15 m")
    axes[1].set_xlabel("斜距 / m")
    axes[1].set_ylabel("沿航向尺度 / m")
    axes[1].set_title("格距不等于远距独立分辨率")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=.2)
    fig.suptitle("依据V2手册与理想匀速计算的示意；不含姿态、阴影及波形解码",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(OUT/"声呐同步波束与时间支持示意.png",dpi=180)
    plt.close(fig)
    print(f"saved analytical schematics in {OUT}")


if __name__=="__main__":
    main()
