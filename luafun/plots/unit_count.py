import altair as alt


def unit_count_through_time():
    with open('/home/setepenre/work/LuaFun/botscpp/unit_size.txt', 'r') as f:
        data = [dict(x=i, y=int(r)) for i, r in enumerate(f.readlines())]

    data = alt.Data(values=data)

    chart = alt.Chart(data).mark_line().encode(
        x='x:Q',
        y='y:Q',
    )

    chart.save('unitcount.png', webdriver='firefox')


def message_size_through_time():
    with open('/home/setepenre/work/LuaFun/botscpp/msg_size.txt', 'r') as f:
        data = [dict(x=i, y=int(r)) for i, r in enumerate(f.readlines())]

    data = alt.Data(values=data)

    chart = alt.Chart(data).mark_line().encode(
        x='x:Q',
        y='y:Q',
    )

    chart.save('msgsize.png', webdriver='firefox')


if __name__ == '__main__':
    unit_count_through_time()
    message_size_through_time()


