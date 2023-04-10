from matplotlib import pyplot as plt

if __name__ == '__main__':
    x = [1000, 5000]
    ref = [1.4, 0.2]

    plt.plot(x, ref, label='reference')

    # slope
    d = (ref[1] - ref[0]) / (x[1] - x[0])
    # intercept
    b = ref[1] - d * x[1]

    x_recons = [100, 10000]
    y_recons = [
        d * x_recons[0] + b,
        d * x_recons[1] + b
    ]

    plt.plot(x_recons, y_recons, label='linear')

    plt.xscale('log')  # uncomment to see the difference

    plt.legend()
    plt.show()
