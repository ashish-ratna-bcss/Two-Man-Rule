import multiprocessing as mp
from multiprocessing.managers import BaseManager, DictProxy
import time

class MyManager(BaseManager): pass

def test_manager():
    m = mp.Manager()
    d = m.dict()
    
    MyManager.register('get_d', callable=lambda: d, proxytype=DictProxy)
    
    server_mgr = MyManager(address=('127.0.0.1', 50002), authkey=b'abc')
    server = server_mgr.get_server()
    
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    
    # Client side
    class MyClientManager(BaseManager): pass
    MyClientManager.register('get_d', proxytype=DictProxy)
    client_mgr = MyClientManager(address=('127.0.0.1', 50002), authkey=b'abc')
    client_mgr.connect()
    
    try:
        d_proxy = client_mgr.get_d()
        d_proxy['test'] = 2
        print(f"Success with proxytype! Value: {d_proxy['test']}")
    except Exception as e:
        print(f"Error with proxytype: {type(e).__name__}: {e}")

if __name__ == '__main__':
    test_manager()
