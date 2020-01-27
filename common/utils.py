import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


# 결정 경계를 그리는 함수
def plot_decision_regions(x, y, classifier, test_idx=None, resolution=0.02):
    # 마커와 컬러맵 설정
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    # 분류할 클래스 개수를 구한다.
    class_count = len(np.unique(y))  # 3개
    # ['red', 'blue', 'lightgreen'] 세개로 컬러맵 생성
    color_map = ListedColormap(colors[:class_count])

    x_min = x[:, 0].min() - 1
    x_max = x[:, 0].max() + 1
    y_min = x[:, 1].min() - 1
    y_max = x[:, 1].max() + 1

    # meshgrid 함수는 축에 해당하는 1차원 배열을 받아서 벡터 공간의 모든 좌표를 담은 행렬을 반환한다.
    # 예를들어 x1, x2 = np.meshgrid([0,1],[2,3])이면, (0,2),(1,2),(0,3),(1,3) 네개 점에 대한
    # x축 값 x1 = [[0,1][0,1]], y축 값 y1 = [[2,2][3,3]]를 반환
    xx, yy = np.meshgrid(
        # arange 함수는 min부터 max까지 resolution만큼씩 커지는 모든 값을 배열로 반환한다.
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    # ravel 함수는 입력된 배열을 1차원으로 펼친다.
    # T는 행렬을 전치해서 두 개의 열이 되도록 바꾼다. 이 두 열이 xy 평면의 좌표값이 된다.
    z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    # reshape 함수는 행렬의 크기를 입력받은 shape과 동일하게 바꿔준다.
    z = z.reshape(xx.shape)

    # contourf 함수로 등고선 그래프를 그린다.
    plt.contourf(
        xx, yy,
        z,
        alpha=0.3,  # 산점도의 점 색깔과 구분되어지게 색상을 투명도 조정함
        cmap=color_map  # 등고선으로 그린 경계 내부를 색칠한다.
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    for idx, c1 in enumerate(np.unique(y)):
        # scatter 함수로 산점도를 그린다. 루프를 돌아서 클래스별로 다른 색과 모양으로 그린다.
        plt.scatter(
            x=x[y == c1, 0],
            y=x[y == c1, 1],
            alpha=0.8,  # 투명도, 클수록 진하다. 0.0으로 하면 안보임
            c=colors[idx],  # 마커의 색상을 결정
            marker=markers[idx],  # 마커의 모양을 결정
            label=c1,  # 해당 모양과 색상이 의미하는 붓꽃 이름
            edgecolor='black'
        )

    # 테스트 샘플을 부각하기 위해서 동그라미를 한번 더 씌운다.
    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]

        plt.scatter(
            x_test[:, 0], x_test[:, 1],
            c='',  # 투명하게 설정
            edgecolor='black',  # 테두리를 검은색으로 설정
            alpha=1.0,
            linewidth=1, marker='o',
            s=100,  # 마커의 크기를 설정
            label='test_set'
        )
