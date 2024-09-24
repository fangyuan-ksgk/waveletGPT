import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D



# Plotting Helper function: Fold up negative frequency signals into positive frequency domain 
def fold_up_negative_freq(signals):
    folded_signals = []
    l = len(signals)
    assert l % 2 != 0, "We should have symmetrical positve and negative frequency signals, together with 0-freq signal adds up to odd number of total signals in DFT"
    for i in range(l//2):
        if i == 0:
            folded_signals.append(signals[l//2+i])
        else:
            folded_signals.append(signals[l//2+i] + signals[l//2-i])
    
    folded_signals = np.array(folded_signals)
    
    return folded_signals

# Discrete Fourier Transform Approximation Animation in 3D 
def dft(time_steps, samples, num_freq, extend_time_steps = None):
    
    ns = np.arange(-num_freq, num_freq+1)
    
    # Coefficients for Complex DFT
    cn = np.trapz(samples[np.newaxis, :] * np.exp(-2j * np.pi * ns[:, np.newaxis] * time_steps[np.newaxis, :]), time_steps)
    
    # Decomposed Signals (with negative frequency signals)
    if extend_time_steps is not None:
        time_steps = extend_time_steps
        
    signals = cn[:, np.newaxis] * np.exp(2j * np.pi * ns[:, np.newaxis] * time_steps[np.newaxis, :])
    
    # Positive Frequency Signals (adding +f and -f together, hope is to cancel out imaginary part)
    positive_freq_signals = fold_up_negative_freq(signals)
    
    # Reconstruction with Fourier Series
    fourier_recon = np.sum(signals, axis=0)
    
    return cn, signals, positive_freq_signals, fourier_recon


def animate_approximation_3d(t, positive_freq_signals, step_function):
    
    num_freq = len(positive_freq_signals)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare lines for each frequency and the sum
    lines = [ax.plot([], [], [], label=f'n={freq}')[0] for freq in range(num_freq)]
    sum_line, = ax.plot([], [], [], 'k-', linewidth=2, label='Sum')
    step_line, = ax.plot([], [], [], 'k--', linewidth=2, label='Step Function')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency (n)')
    ax.set_zlabel('Amplitude')
    ax.set_title('3D Plot of Signals Moving to Frequency 0 Plane')

    # Set consistent view limits
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, num_freq)
    ax.set_zlim(np.min(positive_freq_signals.real) - 0.5, np.max(positive_freq_signals.real) + 0.5)

    # Adjust the viewing angle for better visibility
    ax.view_init(elev=20, azim=45)
    

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        sum_line.set_data([], [])
        sum_line.set_3d_properties([])
        step_line.set_data([], [])
        step_line.set_3d_properties([])
        return lines + [sum_line, step_line]

    def animate(frame):
        # Total number of frames for moving and summing
        total_frames = num_freq * 2
        
        filter_freq = lambda freq: freq % 2 != 0
        
        if frame < num_freq:
            # First phase: show each signal at its frequency
            for freq in range(frame + 1):
                if filter_freq(freq):
                    lines[freq].set_data(t, np.full_like(t, freq))
                    lines[freq].set_3d_properties(positive_freq_signals[freq].real)
        else:
            # Second phase: move signals to frequency 0 and sum
            move_frame = frame - num_freq
            for freq in range(num_freq):
                if filter_freq(freq):
                    current_freq = max(0, freq - move_frame)
                    if current_freq > 0:
                        lines[freq].set_data(t, np.full_like(t, current_freq))
                        lines[freq].set_3d_properties(positive_freq_signals[freq].real)
                    else:
                        lines[freq].set_data([], [])
                        lines[freq].set_3d_properties([])
                
            # Calculate and plot the sum of moved signals
            sum_signal = np.sum(positive_freq_signals[:move_frame+1], axis=0).real
            sum_line.set_data(t, np.zeros_like(t))
            sum_line.set_3d_properties(sum_signal)
        
        # Always show the step function as a dotted line on the frequency 0 plane
        step_line.set_data(t, np.zeros_like(t))
        step_line.set_3d_properties(step_function)
        
        ax.set_title(f'3D Plot of Signals (Frame {frame+1}/{total_frames})')
        return lines + [sum_line, step_line]

    anim = FuncAnimation(fig, animate, init_func=init, frames=num_freq*2, 
                        interval=200, blit=False, repeat=False)
    
    # plt.legend()
    # plt.tight_layout()
    return anim


def draw_circle_with_wave(frame_idx, positive_freq_signals, wave_pts):
    
    K = positive_freq_signals.shape[0]
    signal_pt = positive_freq_signals[:, frame_idx]
    wave_pt = (wave_pts[0][:, frame_idx], wave_pts[1][frame_idx] * np.ones(K))

    # Create the plot
    plt.figure(figsize=(8, 12))

    # Plot circles | Draw circle on the negative side of x-axis suffices (positive circle is the same)
    for k in range(K):
        radius = np.abs(positive_freq_signals[k, 0])
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = radius * np.cos(theta)
        circle_y = radius * np.sin(theta)
        plt.plot(circle_x, circle_y, color='gray', linestyle='--', label='Circle')

    plt.plot(signal_pt.real, signal_pt.imag, color='green', label='Radius')
    plt.scatter(signal_pt.real, signal_pt.imag, color='red', label='Signal Points')
    plt.scatter(wave_pt[0], wave_pt[1], color='blue', label='Wave Points')
    for r_pt, w_pt in zip(signal_pt, zip(*wave_pt)):
        plt.plot([r_pt.real, w_pt[0]], [r_pt.imag, w_pt[1]], linestyle='--', marker='.', alpha=0.3)

    # Set labels and title
    plt.xlabel('Real Part')
    plt.ylabel('Complex Part')
    plt.title('DFT')

    # Add legend
    plt.legend()

    # Ensure equal aspect ratio
    plt.axis('equal')

    # Add grid
    plt.grid(True)

    # Show the plot
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    
def animate_circle_with_wave(pos_signal_pts, wave_pts):
    K = len(pos_signal_pts)
    num_frames = pos_signal_pts.shape[1]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 12))

    # Plot circles | Draw circle on the negative side of x-axis suffices (positive circle is the same)
    for pts in pos_signal_pts:
        ax.plot(pts.real, pts.imag)
    
    radii = [ax.plot([], [], linewidth=1)[0] for _ in range(K)]
    lines = [ax.plot([], [], linewidth=1, linestyle='--', marker='.', markersize=15)[0] for _ in range(K)]
    curves = [ax.plot([], [], linewidth=2)[0] for _ in range(K)]
    
    def animate(frame):
        signal_pt = pos_signal_pts[:, frame]
        wave_pt = (wave_pts[0][:, frame], wave_pts[1][frame] * np.ones(K))
    
        for k in range(K):
            radii[k].set_data([0, signal_pt.real[k]], [0, signal_pt.imag[k]])
            lines[k].set_data([signal_pt.real[k], wave_pt[0][k]], [signal_pt.imag[k], wave_pt[1][k]])
            
            curve_x, curve_y = curves[k].get_data()
            new_x = np.append(curve_x, wave_pt[0][k])
            new_y = np.append(curve_y, wave_pt[1][k])
            curves[k].set_data(new_x, new_y)

        return radii + lines + curves
    
    # Set labels and title
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Complex Part')
    ax.set_title('DFT Animation')

    # Ensure equal aspect ratio
    ax.set_aspect('equal')
    ax.set_ylim(-1.6, 1.5)

    # Add grid
    ax.grid(True)
    
    # Apply tight layout to the axes
    ax.set_tight_layout(True)

    # Create animation
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=20, blit=True)

    return anim 


def draw_circle_with_wave_exp(frame_idx, positive_freq_signals, wave_pts):
    
    K = positive_freq_signals.shape[0]
    signal_pt = positive_freq_signals[:, frame_idx]
    wave_pt = (wave_pts[0][:, frame_idx], wave_pts[1][frame_idx] * np.ones(K))

    # Create the plot
    plt.figure(figsize=(8, 12))

    # Plot circles | Draw circle on the negative side of x-axis suffices (positive circle is the same)
    for k in range(K):
        radius = np.abs(positive_freq_signals[k, 0])
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = radius * np.cos(theta)
        circle_y = radius * np.sin(theta)
        plt.plot(circle_x, circle_y, color='gray', linestyle='--', label='Circle')

    plt.plot(signal_pt.imag, signal_pt.real, color='green', label='Radius')
    plt.scatter(signal_pt.imag, signal_pt.real, color='red', label='Signal Points')
    plt.scatter(wave_pt[1], wave_pt[0], color='blue', label='Wave Points')
    for r_pt, w_pt in zip(signal_pt, zip(*wave_pt)):
        plt.plot([r_pt.imag, w_pt[1]], [r_pt.real, w_pt[0]], linestyle='--', marker='.', alpha=0.3)

    # Set labels and title
    plt.xlabel('Real Part')
    plt.ylabel('Complex Part')
    plt.title('DFT')

    # Add legend
    plt.legend()

    # Ensure equal aspect ratio
    plt.axis('equal')

    # Add grid
    plt.grid(True)

    # Show the plot
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    
def animate_circle_with_wave_exp(pos_signal_pts, wave_pts, speed_up_factor = 1):
    K = len(pos_signal_pts)
    num_frames = pos_signal_pts.shape[1]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot circles | Draw circle on the negative side of x-axis suffices (positive circle is the same)
    for pts in pos_signal_pts:
        ax.plot(pts.real, pts.imag)
    
    radii = [ax.plot([], [], linewidth=1)[0] for _ in range(K)]
    lines = [ax.plot([], [], linewidth=1, linestyle='--', marker='.', markersize=15)[0] for _ in range(K)]
    curves = [ax.plot([], [], linewidth=2)[0] for _ in range(K)]
    
    def animate(frame):

        actual_frame = frame * speed_up_factor
        if actual_frame >= num_frames:
            actual_frame = num_frames - 1

        signal_pt = pos_signal_pts[:, actual_frame]
        wave_pt = (wave_pts[0][:, actual_frame], wave_pts[1][actual_frame] * np.ones(K))
    
        for k in range(K):
            radii[k].set_data([0, signal_pt.imag[k]], [0, signal_pt.real[k]])
            lines[k].set_data([signal_pt.imag[k], wave_pt[1][k]], [signal_pt.real[k], wave_pt[0][k]])
            
            curve_x, curve_y = curves[k].get_data()
            new_x = np.append(curve_x, wave_pt[1][k])
            new_y = np.append(curve_y, wave_pt[0][k])
            curves[k].set_data(new_x, new_y)

        return radii + lines + curves
    
    # Set labels and title
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Complex Part')
    ax.set_title('Discrete Fourier Transform')

    # Ensure equal aspect ratio
    ax.set_aspect('equal')
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(-1.5, 5.6)

    # Add grid
    ax.grid(True)
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=num_frames // speed_up_factor, interval=1, blit=True)

    return anim 


def set_draw_params(k):
    marker_size = max(10, 30 - k * 2)  # Decrease marker size as k increases
    line_width = max(0.3, 2 - k * 0.15)  # Decrease line width as k increases
    alpha = max(0.3, 1 - k * 0.02)  # Decrease opacity as k increases
    
    # Define a list of colors to cycle through
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    color = colors[k % len(colors)]  # Cycle through colors based on k
    
    return marker_size, line_width, alpha, color


def draw_composite_cricle_wavelet(frame_idx, radius_start_pts, radius_end_pts, wave_pts):
        
    K = radius_end_pts.shape[0]

    signal_start_pt = radius_start_pts[:, frame_idx]
    signal_end_pt = radius_end_pts[:, frame_idx]

    wave_pt = (wave_pts[0][frame_idx], wave_pts[1][frame_idx]) # Single Point here | Different from before

    plt.figure(figsize=(12, 8))

    # Plot circles | Draw circle on the negative side of x-axis suffices (positive circle is the same)
    for k in range(K):
        circle_center = signal_start_pt[k]
        radius = np.abs(signal_end_pt[k] - signal_start_pt[k])
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = circle_center.real + radius * np.cos(theta)
        circle_y = circle_center.imag + radius * np.sin(theta)
        
        marker_size, line_width, alpha, color_ = set_draw_params(k)

        
        plt.plot(circle_x, circle_y, color=color_, 
                linewidth=line_width, alpha=alpha, label='Circle' if k == 0 else "")

    # Plot Wave Point and Radii
    for k in range(K):
        marker_size, line_width, alpha, color_ = set_draw_params(k)

        plt.quiver(signal_start_pt[k].real, signal_start_pt[k].imag,
            signal_end_pt[k].real - signal_start_pt[k].real,
            signal_end_pt[k].imag - signal_start_pt[k].imag,
            color=color_, alpha=alpha, scale=1, scale_units='xy', angles='xy',
            width=line_width * 0.0015, headwidth=marker_size * 0.15, 
            headlength=marker_size * 0.15, headaxislength=marker_size * 0.15)


    plt.scatter(wave_pt[0], wave_pt[1], color=color_, label='Wave Points')

    r_pt = signal_end_pt[-1]
    plt.plot([r_pt.real, wave_pt[0]], [r_pt.imag, wave_pt[1]], marker='.')

    # Set labels and title
    plt.xlabel('Real Part')
    plt.ylabel('Complex Part')
    plt.title('DFT')

    # Add legend
    plt.legend()

    # Ensure equal aspect ratio
    plt.axis('equal')

    # Add grid
    plt.grid(True)

    # Show the plot
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    
def animate_composite_circle_wavelet(radius_start_pts, radius_end_pts, wave_pts):
    K = radius_end_pts.shape[0]
    num_frames = radius_end_pts.shape[1]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Initialize empty plots for circles, radii, wave point, and curve
    circles = [ax.plot([], [], alpha=0.5)[0] for _ in range(K)]
    radii = [ax.plot([], [], 'r-')[0] for _ in range(K)]  # Changed from quiver to plot
    wave_point, = ax.plot([], [], 'ro', markersize=5)
    connecting_line, = ax.plot([], [], 'g--')
    curve, = ax.plot([], [], 'b-', linewidth=2)  # Added curve for composite signal

    ax.set_xlim(np.min(radius_start_pts.real) - 1, np.max(wave_pts[0]) + 1)
    ax.set_ylim(np.min(radius_start_pts.imag) - 1, np.max(wave_pts[1]) + 1)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Complex Part')
    ax.set_title('DFT Animation')

    def animate(frame):
        for k in range(K):
            # Update circles
            marker_size, line_width, alpha, color = set_draw_params(k)
            circle_center = radius_start_pts[k, frame]
            radius = np.abs(radius_end_pts[k, frame] - radius_start_pts[k, frame])
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = circle_center.real + radius * np.cos(theta)
            circle_y = circle_center.imag + radius * np.sin(theta)
            circles[k].set_data(circle_x, circle_y)
            circles[k].set_color(color)
            circles[k].set_alpha(alpha)
            circles[k].set_linewidth(line_width)

            # Update radii
            radii[k].set_data([radius_start_pts[k, frame].real, radius_end_pts[k, frame].real],
                              [radius_start_pts[k, frame].imag, radius_end_pts[k, frame].imag])
            radii[k].set_color(color)
            radii[k].set_alpha(alpha)
            radii[k].set_linewidth(line_width)

        # Update wave point and connecting line
        wave_point.set_data(wave_pts[0][frame], wave_pts[1][frame])
        connecting_line.set_data([radius_end_pts[-1, frame].real, wave_pts[0][frame]],
                                 [radius_end_pts[-1, frame].imag, wave_pts[1][frame]])
        connecting_line.set_color(color)
        
        # Update curve (composite signal)
        curve.set_data(wave_pts[0][:frame+1], wave_pts[1][:frame+1])
        curve.set_color(color)
        
        return circles + radii + [wave_point, connecting_line, curve]
        
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=50, blit=True)
    plt.close(fig)  # Prevent duplicate display in Jupyter
    return anim




def animate_composite_circle_wavelet_exp(radius_start_pts, radius_end_pts, wave_pts):
    K = radius_end_pts.shape[0]
    num_frames = radius_end_pts.shape[1]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Initialize empty plots for circles, radii, wave point, and curve
    circles = [ax.plot([], [], alpha=0.5)[0] for _ in range(K)]
    radii = [ax.plot([], [], 'r-')[0] for _ in range(K)]  # Changed from quiver to plot
    wave_point, = ax.plot([], [], 'ro', markersize=5)
    connecting_line, = ax.plot([], [], 'g--')
    curve, = ax.plot([], [], 'b-', linewidth=2)  # Added curve for composite signal

    ax.set_ylim(np.min(radius_start_pts.real) - 1, np.max(wave_pts[0]) + 1)
    ax.set_xlim(np.min(radius_start_pts.imag) - 1, np.max(wave_pts[1]) + 1)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_ylabel('Real Part')
    ax.set_xlabel('Complex Part')
    ax.set_title('DFT Animation')

    def animate(frame):
        for k in range(K):
            # Update circles
            marker_size, line_width, alpha, color = set_draw_params(k)
            circle_center = radius_start_pts[k, frame]
            radius = np.abs(radius_end_pts[k, frame] - radius_start_pts[k, frame])
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = circle_center.real + radius * np.cos(theta)
            circle_y = circle_center.imag + radius * np.sin(theta)
            circles[k].set_data(circle_y, circle_x)
            circles[k].set_color(color)
            circles[k].set_alpha(alpha)
            circles[k].set_linewidth(line_width)

            # Update radii
            radii[k].set_data([radius_start_pts[k, frame].imag, radius_end_pts[k, frame].imag],
                              [radius_start_pts[k, frame].real, radius_end_pts[k, frame].real])
            radii[k].set_color(color)
            radii[k].set_alpha(alpha)
            radii[k].set_linewidth(line_width)

        # Update wave point and connecting line
        wave_point.set_data(wave_pts[1][frame], wave_pts[0][frame])
        connecting_line.set_data([radius_end_pts[-1, frame].imag, wave_pts[1][frame]],
                                 [radius_end_pts[-1, frame].real, wave_pts[0][frame]])
        connecting_line.set_color(color)
        
        # Update curve (composite signal)
        curve.set_data(wave_pts[1][:frame+1], wave_pts[0][:frame+1])
        curve.set_color(color)
        
        return circles + radii + [wave_point, connecting_line, curve]
        
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=50, blit=True)
    plt.close(fig)  # Prevent duplicate display in Jupyter
    return anim