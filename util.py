import subprocess


def run_subprocess(process_args: list):
    stream_p = subprocess.Popen(
        process_args,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    while True:
        output = stream_p.stdout.readline()
        if stream_p.poll() is not None:
            break
        if output:
            print(output.strip())


def show_top(topics, n):
    return sorted(topics[n]["words"].items(), key=lambda k: k[1])[-10:]


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def rank(arr, val):
    # given value 0-1 get nearest index from array
    val = clamp(val, 0, 0.99)
    return arr[int(val * len(arr))]


def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def scale(old_value, old_min, old_max, new_min, new_max, use_clamp=False):
    val = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    if use_clamp:
        return clamp(val, new_min, new_max)
    return val
