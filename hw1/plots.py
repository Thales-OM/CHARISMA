import re
import matplotlib.pyplot as plt

def parse_input(text):
    """
    Parse the input text and extract the numeric values
    """
    pattern = r"(\d+)\s+(\d+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, text)
    data = []
    for match in matches:
        N, P, Time_parallel, Time_sequential, GFlops_parallel, GFlops_sequential = match
        data.append((int(N), int(P), float(Time_parallel), float(Time_sequential), float(GFlops_parallel), float(GFlops_sequential)))
    return data

def plot_charts(data):
    """
    Plot the two line charts for Time and GFlops
    """
    # Separate data by N
    N_values = set(N for N, _, _, _, _, _ in data)
    Time_data = {}
    GFlops_data = {}
    for N in sorted(list(N_values)):
        Time_data[N] = [(P, Time_parallel, Time_sequential) for N_, P, Time_parallel, Time_sequential, _, _ in data if N_ == N]
        GFlops_data[N] = [(P, GFlops_parallel, GFlops_sequential) for N_, P, _, _, GFlops_parallel, GFlops_sequential in data if N_ == N]

    # Plot Time chart
    fig, axs = plt.subplots((len(N_values)+1)//2, 2, figsize=(12, 6*((len(N_values)+1)//2)))
    fig.suptitle("Time (seconds) (M=1000, K=1000)", y=0.92)
    for i, N in enumerate(sorted(list(N_values))):
        Ps, Time_parallel, Time_sequential = zip(*Time_data[N])
        if len(N_values) <= 2:
            axs.plot(Ps, Time_parallel, label="Parallel")
            axs.plot(Ps, Time_sequential, label="Sequential")
            axs.set_title(f"N={N}")
            axs.set_xlabel("Number of threads (P)")
            axs.set_ylabel("Time, sec")
            axs.legend()
        else:
            axs[i//2, i%2].plot(Ps, Time_parallel, label="Parallel")
            axs[i//2, i%2].plot(Ps, Time_sequential, label="Sequential")
            axs[i//2, i%2].set_title(f"N={N}")
            axs[i//2, i%2].set_xlabel("Number of threads (P)")
            axs[i//2, i%2].set_ylabel("Time, sec")
            axs[i//2, i%2].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot GFlops chart
    fig, axs = plt.subplots((len(N_values)+1)//2, 2, figsize=(12, 6*((len(N_values)+1)//2)))
    fig.suptitle("GFlops (M=1000, K=1000)", y=0.92)
    for i, N in enumerate(sorted(list(N_values))):
        Ps, GFlops_parallel, GFlops_sequential = zip(*GFlops_data[N])
        if len(N_values) <= 2:
            axs.plot(Ps, GFlops_parallel, label="Parallel")
            axs.plot(Ps, GFlops_sequential, label="Sequential")
            axs.set_title(f"N={N}")
            axs.set_xlabel("Number of threads (P)")
            axs.set_ylabel("GFlops")
            axs.legend()
        else:
            axs[i//2, i%2].plot(Ps, GFlops_parallel, label="Parallel")
            axs[i//2, i%2].plot(Ps, GFlops_sequential, label="Sequential")
            axs[i//2, i%2].set_title(f"N={N}")
            axs[i//2, i%2].set_xlabel("Number of threads (P)")
            axs[i//2, i%2].set_ylabel("GFlops")
            axs[i//2, i%2].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Read data from console
text = """
    500                   1             2.02797             1.64941            0.493103            0.606278
    500                   2             1.21763             1.71111            0.821268            0.584417
    500                   4            0.673735             1.63732             1.48426            0.610754
    500                   8            0.415814              1.6908             2.40492            0.591436
    500                  16            0.226018             1.67096             4.42443            0.598457
    1000                   1             3.54208             2.89028             0.56464            0.691975
    1000                   2              2.0675             2.88787            0.967353            0.692553
    1000                   4             1.15206             2.90306             1.73602            0.688929
    1000                   8            0.657532             2.88916             3.04168            0.692244
    1000                  16            0.355496             2.88112             5.62594            0.694174
    1500                   1             5.77437             4.78766            0.519537            0.626611
    1500                   2             3.19885             4.71027            0.937838            0.636906
    1500                   4             1.73264             4.63984             1.73146            0.646574
    1500                   8            0.976145             4.63141             3.07331            0.647751
    1500                  16             0.57207             4.68999             5.24411             0.63966
"""

data = parse_input(text)
plot_charts(data)
