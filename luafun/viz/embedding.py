import torch
import torch.linalg


class VizEmbeddingSpace:

    def pca_manual(self, x: torch.Tensor, normalize=None):
        with torch.no_grad():
            # std = x.std(dim=1)
            mean = x.mean(dim=1).unsqueeze(1)

            cov = x.mm(x.T) - mean.mm(mean.T)

            # a, b = torch.eig(cov, True)

            # U, S, Vh = torch.linalg.svd(cov)

            U, S, V = torch.svd(cov, some=False)

            return x.T.mm(V[:, :3]), S[:3].sum() / S.sum()

    def pca_torch(self, x: torch.Tensor, normalize=None):
        U, S, V = torch.pca_lowrank(x, center=False, niter=10)
        return torch.matmul(x, V[:, :3]), None


def main():
    from luafun.train.metrics import MetricWriter
    from luafun.utils.options import datapath
    from luafun.model.components import CategoryEncoder
    import luafun.game.constants as const

    uid = 'f42dedc8acd14f86b8be3520a41195b3'
    writer = MetricWriter(datapath('metrics'), uid)

    hero_encoder = CategoryEncoder(const.HERO_COUNT, 128)

    henc = writer.load_weights('henc')
    hero_encoder.load_state_dict(henc)
    # ==== Done loading the model

    weight = list(hero_encoder.parameters())[0]
    print(weight.shape)

    viz = VizEmbeddingSpace()
    proj, var = viz.pca_torch(weight)
    print(var)

    points = []
    for i in range(122):
        x, y, z = proj[i, :] * 100
        hero = const.HERO_LOOKUP.from_offset(i)
        name = hero.get('pretty_name', hero.get('alias', hero.get('name')))
        points.append(dict(x=x.item(), y=y.item(), z=z.item(), name=name))

    import altair as alt
    alt.themes.enable('dark')

    # .mark_point().encode(
    #     x='x:Q',
    #     y='y:Q',
    #     color='z:Q',
    #     text='name:N'
    # )
    chart = alt.Chart(alt.Data(values=points)).mark_text().encode(
        x='x:Q',
        y='y:Q',
        color='z:Q',
        text='name:N'
    ).properties(
        width=1980 * 0.75,
        height=1080 * 0.75
    ).interactive()
    chart.save('chart_2.html')
    # chart.save('chart.png', webdriver='firefox')





if __name__ == '__main__':
    main()
