import time
import xmlrpc.client


def test():
    server = xmlrpc.client.ServerProxy("http://192.168.137.231:8000")
    print(server.system.listMethods())
    # print(server.remote_set_state("small"))
    time.sleep(2)
    L_res = None
    L_rank = None
    while L_res is None:
        L_rank = server.get_L_rank()
        L_res = server.get_L_result()
        time.sleep(0.4)
    print(f"L: {L_res} : {L_rank}")

    print(server.remote_set_state("big"))
    time.sleep(2)
    D_res = None
    while D_res is None:
        D_res = server.get_D_res()
        time.sleep(0.4)
    print(f"D: {D_res}")


if __name__ == '__main__':
    test()
