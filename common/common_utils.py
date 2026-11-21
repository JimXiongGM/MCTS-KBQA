import argparse
import functools
import json
import os
import pickle
import subprocess
import traceback
from concurrent import futures
from datetime import date, datetime

from loguru import logger
from tqdm import tqdm


def read_json(path="test.json"):
    with open(path, "r", encoding="utf-8") as f1:
        res = json.load(f1)
    return res


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        else:
            return json.JSONEncoder.default(self, obj)


def _set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def save_to_json(obj, path, _print=True):
    if _print:
        print(f"SAVING: {path}")
    if type(obj) == set:
        obj = list(obj)
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f1:
        json.dump(
            obj,
            f1,
            ensure_ascii=False,
            indent=4,
            cls=ComplexEncoder,
            default=_set_default,
        )
    if _print:
        res = subprocess.check_output(f"ls -lh {path}", shell=True).decode(
            encoding="utf-8"
        )
        print(res)


def read_pkl(path="test.pkl", **kwargs):
    with open(path, "rb") as f1:
        res = pickle.load(f1, **kwargs)
    return res


def save_to_pkl(obj, path, _print=True):
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    payload = pickle.dumps(obj)
    with open(path, "wb") as f1:
        f1.write(payload)
    if _print:
        res = subprocess.check_output(f"ls -lh {path}", shell=True).decode(
            encoding="utf-8"
        )
        print(res)


def read_jsonl(path="test.jsonl", desc="", max_instances=None, _id_to_index_key=False):
    with open(path, "r", encoding="utf-8") as f1:
        res = []
        _iter = tqdm(enumerate(f1), desc=desc, ncols=150) if desc else enumerate(f1)
        for idx, line in _iter:
            if max_instances and idx >= max_instances:
                break
            res.append(json.loads(line.strip()))
    if _id_to_index_key:
        id_to_index = {i[_id_to_index_key]: idx for idx, i in enumerate(res)}
        return res, id_to_index
    else:
        return res


def jsonl_generator(path, topn=None, total=None, percent=1, update_func=None, **kwargs):
    """
    usage:
    succ = 0
    def _update():
        return {"success": succ}
    for item in jsonl_generator(path, total=123, update_func=_update):
    """
    total = total if total else file_line_count(path)
    if not total:
        return []
    topn = topn if topn else int(percent * total) + 1
    with open(path) as f1:
        pbar = tqdm(f1, total=min(total, topn), ncols=150, **kwargs)
        for idx, line in enumerate(pbar):
            if idx >= topn:
                break
            yield json.loads(line.strip())
            if update_func:
                info = update_func()
                pbar.set_postfix(ordered_dict=info)


def save_to_jsonl(obj, path, _print=True):
    """
    Object of type set is not JSON serializable. so PAY ATTENTION to data type.
    """
    if isinstance(obj, set):
        obj = list(obj)
    elif isinstance(obj, dict):
        obj = obj.items()
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f1:
        for line in obj:
            f1.write(json.dumps(line, ensure_ascii=False) + "\n")
    if _print:
        res = subprocess.check_output(f"ls -lh {path}", shell=True).decode(
            encoding="utf-8"
        )
        print(res)


def colorful(text, color="yellow"):
    if color == "yellow":
        text = "\033[1;33m" + str(text) + "\033[0m"
    elif color == "grey":
        text = "\033[1;30m" + str(text) + "\033[0m"
    elif color == "green":
        text = "\033[1;32m" + str(text) + "\033[0m"
    elif color == "red":
        text = "\033[1;31m" + str(text) + "\033[0m"
    elif color == "blue":
        text = "\033[1;94m" + str(text) + "\033[0m"
    else:
        pass
    return text


def underline_text(text):
    underline_seq = "\033[4m"  # ANSI escape sequence for underline
    reset_seq = "\033[0m"  # ANSI escape sequence to reset formatting

    return f"{underline_seq}{text}{reset_seq}"


def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data", type=str)
    parser.add_argument("--load_model", action="store_true")
    parser.set_defaults(load_model=True)

    args = parser.parse_args()
    return args


def time_now(fotmat="%Y-%m-%d %H:%M:%S"):
    date_time = datetime.now().strftime(fotmat)
    return date_time


def multi_process(
    items,
    process_function,
    postprocess_function=None,
    total=None,
    cpu_num=None,
    chunksize=1,
    tqdm_disable=False,
    debug=False,
    spawn=False,
    dummy=False,
    unordered=False,
    **kwargs,
):

    if isinstance(items, list):
        total = len(items)

    import functools
    from multiprocessing import cpu_count

    mapper = functools.partial(process_function, **kwargs)

    cpu_num = 1 if debug else cpu_num
    cpu_num = cpu_num or cpu_count()

    # debug
    if debug:
        res = []
        for idx, i in tqdm(enumerate(items), ncols=100):
            r = mapper(i)
            if postprocess_function:
                r = postprocess_function(r)
            res.append(r)
        return res

    if dummy:
        from multiprocessing.dummy import Pool

        pool = Pool(processes=cpu_num)
    else:
        if spawn:
            import torch

            ctx = torch.multiprocessing.get_context("spawn")
            pool = ctx.Pool(processes=cpu_num)
        else:
            from multiprocessing import Pool

            pool = Pool(processes=cpu_num)

    res = []
    _d = "Dummy " if dummy else ""
    pbar = tqdm(
        total=total,
        ncols=100,
        colour="green",
        desc=f"{_d}{cpu_num} CPUs processing",
        disable=tqdm_disable,
    )

    if unordered:
        _func = pool.imap_unordered
    else:
        _func = pool.imap

    # if error, save to some tmp file.
    try:
        for r in _func(mapper, items, chunksize=chunksize):
            if postprocess_function:
                r = postprocess_function(r)
            res.append(r)
            pbar.update()
    except Exception as e:
        traceback.print_exc()
        _time = time_now()
        pickle.dump(res, open(f"error-save {_time}.pkl", "wb"))
        print(f"file save to: error-save {_time}.pkl")

    if res:
        return res


def timeout(seconds):
    executor = futures.ThreadPoolExecutor(1)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            future = executor.submit(func, *args, **kw)
            return future.result(timeout=seconds)

        return wrapper

    return decorator


DEFAULT_CASE_TIMEOUT_SECONDS = 600


def run_thread_pool_cases(
    items,
    fn,
    *,
    max_workers: int,
    case_timeout_seconds: float | None = DEFAULT_CASE_TIMEOUT_SECONDS,
    desc: str = "Running",
    item_id_fn=None,
    collect_results: bool = False,
    ncols: int = 100,
):
    """Run fn(item) in a thread pool; skip cases that exceed case_timeout_seconds.

    Keep at most max_workers cases submitted at once. This makes the timeout measure
    actual running time instead of time spent waiting in an executor queue, and it
    avoids flooding local LLM / KB API services on full runs.
    """
    import time as _time

    if item_id_fn is None:
        item_id_fn = lambda item: (
            item.get("id", "?") if isinstance(item, dict) else "?"
        )

    total = len(items)
    empty_stats = {"total": 0, "success": 0, "timeouts": 0, "errors": 0}
    if total == 0:
        return ([], empty_stats) if collect_results else empty_stats

    results = []
    successes = 0
    timeouts = 0
    errors = 0
    completed = 0

    def _finish(future, item, status, result=None, exc=None):
        nonlocal completed, successes, timeouts, errors
        completed += 1
        item_id = item_id_fn(item)
        if status == "success":
            successes += 1
            if collect_results:
                results.append(result)
        elif status == "timeout":
            timeouts += 1
            logger.warning(
                f"[{completed}/{total}] timeout id={item_id} "
                f"after {case_timeout_seconds}s, skip"
            )
        else:
            errors += 1
            logger.exception(f"[{completed}/{total}] failed id={item_id}: {exc}")

    executor = futures.ThreadPoolExecutor(max_workers=int(max_workers))
    try:
        item_iter = iter(items)
        future_to_item = {}
        submit_time = {}
        pending = set(future_to_item.keys())
        timed_out_running = set()
        submitted_count = 0
        pbar = tqdm(total=total, desc=desc, ncols=ncols) if desc else None

        def _submit_next():
            nonlocal submitted_count
            try:
                item = next(item_iter)
            except StopIteration:
                return False
            future = executor.submit(fn, item)
            future_to_item[future] = item
            submit_time[future] = _time.time()
            pending.add(future)
            submitted_count += 1
            return True

        for _ in range(min(int(max_workers), total)):
            _submit_next()

        wait_timeout = 1.0
        if case_timeout_seconds is not None:
            wait_timeout = min(1.0, max(0.1, case_timeout_seconds / 4))

        try:
            while pending or timed_out_running or submitted_count < total:
                active = pending | timed_out_running
                if not active:
                    _submit_next()
                    continue
                done, _ = futures.wait(
                    active,
                    timeout=wait_timeout,
                    return_when=futures.FIRST_COMPLETED,
                )
                now = _time.time()

                for future in list(done):
                    if future in timed_out_running:
                        timed_out_running.remove(future)
                        future_to_item.pop(future, None)
                        submit_time.pop(future, None)
                        _submit_next()
                        continue

                    if future not in pending:
                        continue

                    pending.remove(future)
                    item = future_to_item.pop(future)
                    submit_time.pop(future, None)
                    try:
                        result = future.result()
                        _finish(future, item, "success", result=result)
                    except Exception as exc:
                        _finish(future, item, "error", exc=exc)
                    if pbar is not None:
                        pbar.update(1)
                    _submit_next()

                for future in list(pending):
                    if future in done:
                        continue

                    if (
                        case_timeout_seconds is not None
                        and now - submit_time[future] > case_timeout_seconds
                    ):
                        pending.remove(future)
                        item = future_to_item[future]
                        _finish(future, item, "timeout")
                        if pbar is not None:
                            pbar.update(1)
                        if future.cancel():
                            future_to_item.pop(future, None)
                            submit_time.pop(future, None)
                            _submit_next()
                        else:
                            timed_out_running.add(future)
        except KeyboardInterrupt:
            logger.warning("\nKeyboard interrupt detected. Cancelling pending tasks...")
            for future in pending | timed_out_running:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if pbar is not None:
                pbar.close()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    stats = {
        "total": total,
        "success": successes,
        "timeouts": timeouts,
        "errors": errors,
    }
    if collect_results:
        return results, stats
    return stats


def wc_l(path):
    try:
        res = subprocess.check_output(f"wc -l {path}", shell=True).decode(
            encoding="utf-8"
        )
        line_num = int(res.split()[0])
    except Exception as e:
        line_num = None
    return line_num


# @timeout(10)
def file_line_count(path):
    return wc_l(path)


from urllib import parse, request


def send_meg_to_wx(text="default text", desp="default content", sendkey=None):
    """
    Send message to wx
    echo "# server key
    export sctapi_sendkey=xxxx
    " >> ~/.bashrc && source ~/.bashrc
    Based on ServerChan: https://sct.ftqq.com/
    """
    sendkey = sendkey or os.environ.get("sctapi_sendkey", None)
    if sendkey is None:
        return
    query_args = {"text": text, "desp": desp}

    try:
        r = request.Request(
            url=f"https://sctapi.ftqq.com/{sendkey}.send",
            data=parse.urlencode(query_args).encode("utf-8"),
        )
        response = request.urlopen(r, timeout=10).read().decode("utf-8")
        # print(response)
    except:
        pass


def is_vscode_debug_mode():
    import sys

    return sys.breakpointhook != sys.__breakpointhook__


def time_diff_to_str(time_diff):
    hours = int(time_diff // 3600)
    minutes = int((time_diff % 3600) // 60)
    seconds = int(time_diff % 60)
    time_diff_str = f"{hours} hours {minutes} minutes {seconds} seconds"
    return time_diff_str


def exception_logger(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                # Ensure directory exists
                os.makedirs("log_exception", exist_ok=True)

                # Formulate the log filename
                log_filename = f"log_exception/{name}.log"

                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Retrieve the traceback information
                tb = traceback.extract_tb(ex.__traceback__)

                # Extract the relevant data from the traceback
                file_name, line_number, function_name, text = tb[-1]

                # Collect log messages
                log_messages = [
                    f"{timestamp} ERROR Exception occurred in function '{func.__name__}':",
                    f"{timestamp} ERROR Exception type: {type(ex).__name__}",
                    f"{timestamp} ERROR Exception args: {ex.args}",
                    f"{timestamp} ERROR Occurred at: {file_name}, line {line_number}, in {function_name}",
                    f"{timestamp} ERROR Code context: {text}",
                    f"{timestamp} ERROR Function arguments: args={args}, kwargs={kwargs}",
                    "",
                ]

                # Write log messages to the file
                with open(log_filename, "a") as f:
                    for message in log_messages:
                        print(message, file=f)

                # Optionally re-raise the exception if you want to handle it further up
                raise

        return wrapper

    return decorator


import functools


def catch_keyboard_interrupt(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logger.warning("\nKeyboard interrupt detected. Exiting immediately...")
            os._exit(130)

    return wrapper


if __name__ == "__main__":
    send_meg_to_wx(text="test", desp="test")
