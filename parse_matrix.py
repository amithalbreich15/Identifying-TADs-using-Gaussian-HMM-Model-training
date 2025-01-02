import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import hmmlearn as hmm
from hmmlearn.hmm import GaussianHMM
import hmmlearn.hmm
import random
import matplotlib.colors as mcolors

from plotly.express import pd
from sklearn.decomposition import PCA
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

N_COMPONENTS = 3
GENOME_START_IDX = 0
SEED = np.random.seed(42)


class ConstrainedGaussianHMM(GaussianHMM):
    def _do_mstep(self, stats):
        # do the standard HMM learning step
        super()._do_mstep(stats)

        # find which states are where
        # we name them based on their typical value
        # m = np.squeeze(self.means_)
        # s = np.argsort(m)
        s = [1, 0, 2]
        s1, s2, s3 = s

        # constrain the transition matrix
        # disallow s1->s2 and/or s2->s3 is not allowed
        self.transmat_[s1, s2] = 0.0
        self.transmat_[s2, s3] = 0.0


def load_hic_matrix(file_path):
    """
    Load HiC Results Matrix from a given file path.

    Parameters:
        file_path (str): Path to the file containing the HiC results matrix.

    Returns:
        numpy.ndarray: HiC Results Matrix
    """
    # Read the file
    matrix_data = np.loadtxt(file_path, delimiter="\t")
    # Check the shape of the matrix_data
    print("Shape of matrix_data:", matrix_data.shape)
    return matrix_data


def draw_hic_heatmap(contact_matrix, colorscale='reds'):
    """
    Draw HiC Results Heatmap from a given matrix.

    Parameters:
        contact_matrix (numpy.ndarray): HiC Results Matrix
        colorscale (str): Name of the colorscale to use for the heatmap.
                          Default is 'reds'.

    Returns:
        None (displays the plot)
    """
    # Take only the upper triangle of the matrix
    upper_triangle = np.triu(contact_matrix)

    # Transpose the matrix to plot "chr2" as a function of "Bins"
    transposed_data = upper_triangle.T

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(z=transposed_data, colorscale=colorscale))

    # Update layout
    fig.update_layout(
        title='HiC Results Heatmap',
        xaxis=dict(title='Bins'),
        yaxis=dict(title='chr2'),
        coloraxis_colorbar=dict(title='HiC Counts')
    )

    # Show plot
    fig.show()

    # Create heatmap
    fig2 = go.Figure(data=go.Heatmap(z=contact_matrix, colorscale='Reds'))

    # Update layout
    fig2.update_layout(
        title='HiC Results Heatmap',
        xaxis=dict(title='Bins'),
        yaxis=dict(title='Bins'),
        coloraxis_colorbar=dict(title='HiC Counts')
    )

    # Show plot
    fig2.show()


def plot_heatmap_using_pyplot(contact_matrix, title='HiC Results Heatmap',
                              cmap='Reds'):
    """
    Plot a heatmap from a given data matrix.

    Parameters:
        contact_matrix (numpy.ndarray): Data matrix for the heatmap.
        title (str): Title of the heatmap. Default is 'HiC Results Heatmap'.
        cmap (str): Colormap for the heatmap. Default is 'Reds'.

    Returns:
        None (displays the plot)
    """
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(contact_matrix, cmap=cmap)
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('Bins')
    # Add colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('HiC Counts')
    plt.show()
    plt.show()


def compute_score_to_bottom(contact_matrix, row_index):
    # Calculate the sum of 50 values above and below the specified cell
    top_sum = np.sum(contact_matrix[max(0, row_index - 50):row_index,
                     row_index + 1])
    bottom_sum = np.sum(contact_matrix[row_index + 1:min(row_index + 51,
                                                         contact_matrix.shape[
                                                             0]),
                        row_index + 1])

    # Calculate the score as the difference between the top and bottom sums
    score = top_sum - bottom_sum
    return score


def compute_score(contact_row, row_index):
    # Calculate the sum of 50 values to the left and right of the specified row_index
    left_sum = np.sum(
        contact_row[max(0, row_index - 50):row_index])
    right_sum = np.sum(contact_row[row_index + 1:min(row_index + 51,
                                                     contact_row.shape[
                                                         0]) - 1])
    # Calculate the score as the difference between the left and right sums
    normalized_score = compute_DI_score(left_sum, right_sum)
    return normalized_score


# def calculate_insulation_score(matrix, bin_size, square_size):
#     # Calculate the dimensions of the matrix
#     matrix_size = len(matrix)
#
#     # Calculate the number of bins
#     num_bins = matrix_size // bin_size
#
#     # Calculate the number of bins in the square
#     num_bins_in_square = square_size // bin_size
#
#     # Initialize a list to store insulation scores
#     insulation_scores = []
#
#     # Iterate over each bin
#     for i in range(num_bins):
#         # Skip bins within 500kb of matrix start/end
#         if i < num_bins_in_square // 2 or i > num_bins - num_bins_in_square // 2:
#             continue
#
#         # Calculate the indices of the square around the current bin
#         start_index = max(0, i - num_bins_in_square // 2)
#         end_index = min(num_bins, i + num_bins_in_square // 2 + 1)
#
#         # Initialize a list to store interaction counts within the square
#         interactions_within_square = []
#
#         # Iterate over each row in the square
#         for j in range(start_index, end_index):
#             # Extract interactions within the square
#             interactions_within_square.extend(
#                 matrix[j, max(0, i - num_bins_in_square // 2):min(num_bins,
#                                                                   i + num_bins_in_square // 2 + 1)])
#
#         # Calculate the mean signal within the square
#         mean_signal = np.mean(interactions_within_square)
#
#         # Assign the mean signal to the current bin
#         insulation_scores.append(mean_signal)
#
#     # Normalize insulation scores
#     mean_insulation = np.mean(insulation_scores)
#     normalized_insulation_scores = [np.log2(insulation_score / mean_insulation)
#                                     for insulation_score in
#                                     insulation_scores]
#
#     return normalized_insulation_scores


def compute_DI_score(b, a):
    if b == 0 and a == 0:
        return 0.0
    sum_avg = (b + a) / 2
    term_1 = (b - a) / (abs(b - a))
    term_2 = ((b - sum_avg) ** 2 + (a - sum_avg) ** 2) / \
             sum_avg
    final_score = term_1 * term_2
    return final_score


# def train_hmm_model(contact_matrix):
#     # Calculate scores for upper triangle
#     scores = [compute_score(contact_matrix[i, :], i) for i in range(
#         contact_matrix.shape[0])]
#     observations = np.array(scores).reshape(-1, 1)
#     model = ConstrainedGaussianHMM(
#         n_components=N_COMPONENTS)  # forward, backward, same# Check for NaN values and
#     # model.startprob_ = np.array([0.5, 0, 0.5])
#     model.init_params = 'smc'
#     model.transmat_ = np.array([[0.8, 0.001, 0.199],
#                                 [0.001, 0.8, 0.199], [0.45, 0.45, 0.1]
#                                 ])
#     # replace them with zeros
#     # observations = np.nan_to_num(observations, nan=0.0)
#     model.fit(observations)
#     predicted_states = model.predict(observations)
#     output_path = 'predicted_seq.txt'
#     means = model.means_
#     posteriors_vec = model.predict_proba(observations)
#     posterior_list = list(posteriors_vec)
#
#     transition_points = find_state_transition_points_with_posterior(
#         posteriors_vec)
#     covariances = model.covars_
#     write_hmm_predicted_sequence(predicted_states, means, covariances,
#                                  transition_points,
#                                  output_path)
#     print("Predicted state sequence:", predicted_states)
#     print(predicted_states.shape)
#     return predicted_states, model, scores, means, covariances, transition_points


def train_hmm_model(contact_matrix, score_method, bin_size=400, square_size=1,
                    score_method_str="di_score"):
    # Calculate scores for upper triangle
    scores = []
    if score_method_str == 'di_score':
        scores = [score_method(contact_matrix[i, :], i) for i in range(
            contact_matrix.shape[0])]
    elif score_method_str == 'insulation_score':
        scores = calculate_insulation_scores(contact_matrix, bin_size,
                                             square_size)
    elif score_method_str == 'insulation_weighted_score':
        scores = calculate_weighted_insulation_score(contact_matrix, bin_size,
                                                     square_size)
    observations = np.array(scores).reshape(-1, 1)
    model = GaussianHMM(n_components=N_COMPONENTS,
                        random_state=42)  # Set the random state for reproducibility
    model.init_params = ''
    # model.startprob_ = np.array([0.5, 0, 0.5])'
    # Initialize start probabilities to something meaningful if known
    # For example, assuming the start is always the Middle state, index 1
    model.startprob_ = np.array(
        [0.0, 0.5, 0.5])  # M state has 100% probability at the start

    # initial_means = np.array([[0],  # Mean close to 0 for the Middle state
    #                           [10],  # Positive mean for the Backward state
    #                           [-10]])  # Negative mean for the Forward state
    # initial_covars = np.array([[[1]],  # Small variance for the Middle state
    #                            [[10]],
    #                            # Larger variance for the Backward state
    #                            [[
    #                                 10]]])  # Larger variance for the Forward state

    # Set initial means and covariances based on prior knowledge
    # model.means_ = np.array(
    #     [[0], [10], [-10]])  # Middle, Backward, Forward state means
    # model.covars_ = np.stack((np.array([[1]]), np.array([[100]]),
    #                           np.array([[100]])))  # Example covariances

    # Initialize the transition matrix
    # This matrix defines the probabilities of moving from one state to another
    # F->M, M->B transitions are allowed; B->M, M->F are disallowed (set to 0)
    model.transmat_ = np.array([
        [0.5, 0.5, 0.0],  # F can transition to F or M
        [0.0, 0.5, 0.5],  # M can transition to M or B
        [0.5, 0.0, 0.5],  # B transitions to B or M
    ])
    # replace them with zeros
    observations = np.nan_to_num(observations, nan=0.0)
    model.fit(observations)
    predicted_states = model.predict(observations)
    output_path = 'predicted_seq.txt'
    means = model.means_
    posteriors_vec = model.predict_proba(observations)
    posterior_list = list(posteriors_vec)

    transition_points = find_state_transition_points_with_posterior(
        posteriors_vec)
    covariances = model.covars_
    write_hmm_predicted_sequence(predicted_states, means, covariances,
                                 transition_points,
                                 output_path)
    print("Predicted state sequence:", predicted_states)
    print(predicted_states.shape)
    print("Score method:", score_method_str)
    return predicted_states, model, scores, means, covariances, transition_points


# Function to plot the distribution graph of the HMM Model Markovian Emissions
def plot_hmm_model_distribution(model):
    means = model.means_
    covars = model.covars_

    # Plot the data generated by the HMM model
    fig, axs = plt.subplots(N_COMPONENTS, sharex=True, sharey=True)
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    # For each hidden state
    for i, (mean, covar, color) in enumerate(
            zip(means, covars, colors[:N_COMPONENTS])):
        # Generate data for this state
        data = np.random.multivariate_normal(mean.ravel(), covar, size=500)

        # Plot this data
        axs[i].hist(data, bins=30, density=True, color=color, alpha=0.5)
        axs[i].set_title(f"Hidden State {i + 1}")

    plt.xlabel("Feature values")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    return plt


def write_hmm_predicted_sequence(predicted_states, means, covs,
                                 transition_points, output_path):
    with open(output_path, 'w+') as f:
        f.write('Sequence Predictions\n')
        f.write(str(list(predicted_states)) + '\n')
        f.write('\nMeans:\n')
        f.write(str(means) + '\n')
        f.write('\nCovariances:\n')
        f.write(str(covs) + '\n')
        f.write('\nTransition Points:\n')
        f.write(str(transition_points) + '\n')


def find_state_transition_points_with_posterior(posteriors_vec):
    # Analyze posterior probabilities to identify transition points
    threshold = 0.3  # Adjust as needed
    transition_points = []
    for i in range(len(posteriors_vec) - 1):
        prob_1 = posteriors_vec[i][0]
        prob_2 = posteriors_vec[i + 1][0]
        if abs(prob_1 - prob_2) > threshold:
            # Check if the difference in probabilities is above threshold
            transition_points.append(i)
    return transition_points


# def predict_tads(states):
#     filter_window = [1, 1, 1, 9999, 9999, 9999, 9999, 9999, 9999, 9999, -1,
#                      -1, -1]
#     conv_output = np.convolve(states, filter_window, mode='same')
#     boundaries = np.where(conv_output < 0)[0]
#     return boundaries


def find_transition_indices(states):
    transition_indices = []
    state_0_counter = 0
    state_1_counter = 0
    state_2_counter = 0
    initial_index = 0
    mem_idx = 0
    transition_point = 0
    for k in range(len(states)):
        if states[k] == 2:
            initial_index = k
            break
    for i in range(initial_index, len(states) - 1):
        if states[i] == 2:
            state_2_counter += 1 and mem_idx + 1 == i
            mem_idx = i
            continue
        if states[i] == 1 and mem_idx + 1 == i:
            state_1_counter += 1
            mem_idx = i
            # i += 1
            continue
        if states[i] == 2 and mem_idx + 1 == i:
            state_2_counter += 1
            mem_idx = i
            transition_point = i
            # i += 1
            continue
        if state_2_counter >= 5 and state_0_counter >= 5:
            transition_indices.append(transition_point)

    return np.array(transition_indices)


def plot_scores_and_boundaries(scores, boundaries):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # bar_labels = ['red', 'blue', 'orange']
    # bar_colors = ['tab:red', 'tab:blue', 'tab:orange']
    score_colors = ['green' if score >= 0 else 'red' for score in scores]

    # Plot scores in the first subplot
    ax1.bar(np.arange(len(scores)), np.squeeze(scores), color=score_colors)

    # Plot boundaries in the second subplot

    ax2.axhline(y=1, color='r', linestyle='-')
    ax2.scatter(boundaries[np.newaxis, :], np.ones((1, boundaries.shape[0])),
                c='b')
    print(f"boundaries: {boundaries}")

    # Set titles and labels
    ax1.set_title('TAD Boundaries')
    ax1.set_xlabel('Interval Index')
    ax1.set_ylabel('Score')

    ax2.set_xlabel('Bin index')
    ax2.set_xticks(range(len(scores)))

    # Show the plot
    fig.tight_layout()
    fig.show()


def predict_tads(states, start_idx, end_idx):
    # Backward value move from 0 to 1
    # M value move from 1 to 0
    boundaries = []
    for i in range(len(states)):
        if states[i] == 2 and np.all(
                states[i:min(states.shape[0], i + 2)] == 2) and np.all(
            states[max(0, i - 2):i] == 1):
            boundaries.append(start_idx + i)
    return np.array(boundaries)


def find_tad_boundaries(states, start_idx, end_idx, min_stable_bins=2):
    # Initialize an array of zeros with the size of end_idx - start_idx
    tad_boundaries = np.zeros(end_idx - start_idx, dtype=int)

    # Initialize a counter for the number of stable bins
    stable_bins_counter = 0
    # Loop through the states, excluding the edges defined by the stability requirement
    for i in range(min_stable_bins, len(states) - min_stable_bins):
        # Increment the stable bin counter if the current state is 1
        if states[i] == 1:
            stable_bins_counter += 1
        else:
            stable_bins_counter = 0  # Reset counter if the state is not 1

        # Check if the state transition from 1 to 2 occurs after a stable region
        if stable_bins_counter >= min_stable_bins and states[i + 1] == 2:
            # Check if the following state 2 is also stable
            if all(states[j] == 2 for j in range(i + 2, i + min_stable_bins + 2)):
                tad_boundaries[i + 1] = 1  # Set the boundary
                # index to 1
                # Skip the checked bins to prevent recounting the same boundary
                i += min_stable_bins

    print(tad_boundaries)
    return tad_boundaries


def predict_tads_with_flags(states, start_idx, end_idx):
    """
    Predict the TAD boundaries and return an array with True flags where boundaries are found.

    Parameters:
    - states (np.array): Array of predicted states
    - start_idx (int): Starting index for prediction
    - end_idx (int): Ending index for prediction

    Returns:
    - boundary_flags (np.array): Array of boolean flags indicating the presence of a boundary
    """

    def has_streak_of_4(states, index):
        streak = states[index:index + 4]
        return np.all(np.isin(streak, [1, 2]))

    def has_streak_of_4_backwards(states, index):
        streak = states[index - 4:index]
        return np.all(np.isin(streak, [1]))

    # Initialize an array with False flags
    boundary_flags = np.full((end_idx - start_idx), 0)

    for i in range(len(states)):
        if states[i] == 2 and (np.all(
                states[i:min(states.shape[0], i + 2)] == 2) and np.all(
            states[max(0, i - 2):i] == 1) or has_streak_of_4_backwards(
            states, i)):
                boundary_flags[i] = 1

    print(boundary_flags)
    print(len(boundary_flags))
    return boundary_flags


def calculate_emission_probabilities(predicted_states):
    """
    Calculate the emission probabilities of given predicted states.

    Parameters:
    - predicted_states (np.array): Array of predicted states for each observation

    Returns:
    - emission_probabilities (dict): Dictionary containing the emission probabilities for each state
    """
    predicted_states = np.array(predicted_states)

    # Calculate the frequency of each state in the predicted states
    state_counts = np.bincount(predicted_states)
    total_states = predicted_states.size

    # Calculate the emission probability for each state
    emission_probabilities = {state: count / total_states for state, count in
                              enumerate(state_counts)}

    # Adjust probabilities to ensure they sum to 1.0 (due to possible rounding errors)
    prob_sum = sum(emission_probabilities.values())
    if prob_sum != 1.0:
        # Normalize to sum to 1.0
        emission_probabilities = {state: prob / prob_sum for state, prob in
                                  emission_probabilities.items()}

    return emission_probabilities


def plot_compartment_vector(compartment):
    # Define colors for compartments A and B
    compartment_colors = {'A': 'red', 'B': 'green'}

    # Create a scatter plot for the compartment vector
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(compartment)), compartment,
                c=[compartment_colors[c] for c in compartment])
    plt.title('Compartment Vector')
    plt.xlabel('Bin #')
    plt.ylabel('Compartment')
    plt.grid(True)
    plt.show()


def plot_scores_comparison_graphs(scores1, scores2, cell_name='Cortex',
                                  start_idx=0, end_idx=100):
    start_idx_x = start_idx * 40 * 1000  # Converting start_idx to kilobase
    end_idx_x = end_idx * 40 * 1000  # Converting end_idx to kilobase

    # Plotting the graph with bins colored based on their values
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i, score in enumerate(scores1):
        if score >= 0:
            ax1.bar((i + start_idx), score, color='red')
        else:
            ax1.bar((i + start_idx), score, color='green')

    for i, score in enumerate(scores2):
        if score >= 0:
            ax2.bar((i + start_idx), score, color='red')
        else:
            ax2.bar((i + start_idx), score, color='green')

    # Centralize the start of the y-axis on 0
    max_abs_score1 = max(abs(np.array(scores1)))
    max_abs_score2 = max(abs(np.array(scores2)))
    max_score = max(max_abs_score1, max_abs_score2)

    ax1.set_ylim(-max_abs_score1, max_abs_score1)
    ax1.set_title(f'Scores Results - {cell_name}')
    ax1.set_ylabel('Scores')
    ax1.set_xlabel('Chromosomal Position [bp]')
    ax1.grid(True)

    ax2.set_ylim(-max_score, max_score)
    ax2.set_ylabel('Scores')
    ax2.set_xlabel('Chromosomal Position [bp]')
    ax2.grid(True)

    # Set x-tick labels to display the desired values
    ax1.set_xticklabels(
        [(i + start_idx) * 40 * 1000 for i in ax1.get_xticks()])
    ax2.set_xticklabels(
        [(i + start_idx) * 40 * 1000
         for i in ax2.get_xticks()])

    fig.tight_layout()
    plt.show()


def plot_hic_pyramids(contact_matrix, start_idx=0):
    plt.figure(figsize=(8, 6))
    upper_triangular = np.triu(contact_matrix, k=1)
    i, j = np.where(upper_triangular)
    r2 = upper_triangular[i, j]
    y = (j - i) / 2
    df = pd.DataFrame({'i': i + y, 'j': y, 'r2': r2})
    norm = mcolors.Normalize(vmin=df['r2'].min(), vmax=df['r2'].max())
    plt.scatter(df['i'], df['j'], c=df['r2'], cmap='Reds', norm=norm,
                alpha=0.7, s=50, edgecolors='none')
    plt.colorbar(label='r2')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().set_yticks([])
    plt.ylabel(None)
    plt.xlabel('Position')
    plt.show()


def plot_scores_pyramids_comparison_graphs(scores1, scores2, contact_matrix,
                                           cell_name='Cortex', start_idx=0):
    # Plotting the graph with bins colored based on their values
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={
        'height_ratios': [2, 2, 2], 'hspace': 0.2})

    # Plot contact matrix pyramid
    upper_triangular = np.triu(contact_matrix, k=1)
    i, j = np.where(upper_triangular)
    r2 = upper_triangular[i, j]
    y = (j - i) / 2
    df = pd.DataFrame({'i': i + y, 'j': y, 'r2': r2})
    norm = mcolors.Normalize(vmin=df['r2'].min(), vmax=df['r2'].max())
    im = ax1.scatter(df['i'], df['j'], c=df['r2'], cmap='Reds', norm=norm,
                     alpha=1.0, s=25, edgecolors='none')
    # cbar = fig.colorbar(im, ax=ax1, label='Normalized Interacting Counts',
    #                     orientation='horizontal',
    #                     pad=0.1)  # Position colorbar horizontally above ax1
    # cbar.ax.xaxis.set_ticks_position('bottom')  # Move colorbar ticks to top

    ax1.set_aspect(
        'auto')  # Let's make the aspect ratio automatic for the first subplot
    ax1.set_yticks([])
    ax1.set_ylabel(None)
    ax1.set_xlabel('Chromosomal Position [bp]')
    ax1.set_title('Hi-C Matrix')

    # Plot scores graphs
    for i, score in enumerate(scores1):
        if score >= 0:
            ax2.bar((i + start_idx), score, color='red')
        else:
            ax2.bar((i + start_idx), score, color='green')

    for i, score in enumerate(scores2):
        if score >= 0:
            ax3.bar((i + start_idx), score, color='green')
        else:
            ax3.bar((i + start_idx), score, color='green')

    # Centralize the start of the y-axis on 0
    max_abs_score1 = max(abs(np.array(scores1)))
    max_abs_score2 = max(abs(np.array(scores2)))
    max_score = max(max_abs_score1, max_abs_score2)

    ax2.set_ylim(-max_abs_score1, max_abs_score1)
    ax2.set_ylabel('Scores')
    ax2.set_xlabel('Chromosomal Position [bp]')
    ax2.set_title('DI Score')
    ax2.grid(True)

    ax3.set_ylim(-max_score, max_score)
    ax3.set_ylabel('Scores')
    ax3.set_xlabel('Chromosomal Position [bp]')
    ax3.set_title('Insulation Score')
    ax3.grid(True)

    # Set x-tick labels to display the desired values
    ax1.set_xticklabels(
        [(i + start_idx) * 40 * 1000 for i in ax2.get_xticks()])
    ax2.set_xticklabels(
        [(i + start_idx) * 40 * 1000 for i in ax2.get_xticks()])
    ax3.set_xticklabels(
        [(i + start_idx) * 40 * 1000 for i in ax3.get_xticks()])

    fig.suptitle(f'Score Methods Comparison - {cell_name}',
                 fontsize=16)  # Add title to the whole figure
    fig.tight_layout()
    plt.show()


def group_scores(scores, is_avg=True):
    num_intervals = len(scores) // 25
    residual = len(scores) % 25

    # Group scores into intervals of 25 values
    interval_sums = []
    for i in range(num_intervals):
        interval = scores[i * 25: (i + 1) * 25]
        if is_avg:
            interval_sum = np.average(interval)
        else:
            interval_sum = np.sum(interval)
        interval_sums.append(interval_sum)

    # If there's a residual, add it to the last interval
    if residual:
        last_interval = scores[num_intervals * 25:]
        if is_avg:
            interval_sum = np.average(last_interval)
        else:
            interval_sum = np.sum(last_interval)
        interval_sums.append(interval_sum)

    return interval_sums


def plot_scores_graph(scores):
    # Define colors for positive and negative values
    colors = ['green' if score >= 0 else 'red' for score in scores]

    # Plotting the graph with bins colored based on their values
    plt.figure(figsize=(10, 6))
    for i, score in enumerate(scores):
        if score >= 0:
            plt.bar(i, score, color='green')
        else:
            plt.bar(i, score, color='red')

    # Centralize the start of the y-axis on 0
    max_abs_score = max(abs(score) for score in scores)
    plt.ylim(-max_abs_score, max_abs_score)

    plt.title('Scores Results')
    plt.xlabel('Interval Index')
    plt.ylabel('Sum of Scores')

    # Set xticks with jumps of 20
    plt.xticks(range(0, len(scores), 20))

    # plt.grid(True)
    plt.show()


def plot_scores_intervals(scores):
    # Calculate the number of intervals
    num_intervals = len(scores) // 100

    # Plot scores for each interval
    for i in range(num_intervals):
        start_index = i * 100
        end_index = min((i + 1) * 100, len(scores))
        interval_scores = scores[start_index:end_index]
        plot_scores_graph(interval_scores)

    # Plot the residual part if the indices aren't divisible by 100
    if len(scores) % 100 != 0:
        residual_scores = scores[num_intervals * 100:]
        plot_scores_graph(residual_scores)


def plot_scores_states(scores, states, start_idx, end_idx):
    # Check if start_idx and end_idx are within bounds
    if start_idx < 0:
        print("Error: start_idx or end_idx out of bounds")
        return

    # Define colors for states
    # states = reorganize_states(states)
    state_colors = [
        'purple' if state == 1 else 'green' if state == 0 else 'red' for state
        in states]

    # Define colors for positive and negative values
    score_colors = ['red' if score >= 0 else 'green' for score in scores]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot scores in the first subplot
    ax1.bar(np.arange(start_idx, end_idx), scores,
            color=score_colors)

    # Plot states in the second subplot
    ax2.scatter(np.arange(start_idx, end_idx), states,
                c=state_colors)

    # Set titles and labels
    ax1.set_title('Scores Results')
    ax1.set_xlabel('HMM States')
    ax1.set_ylabel('Sum of Scores')
    ax1.grid(True)

    ax2.set_xlabel('Bin #')
    ax2.set_ylabel('State')
    ax2.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()


def normalize_insulation_scores(insulation_scores):
    mean_insulation_score = np.mean(insulation_scores)
    normalized_insulation_scores = insulation_scores / mean_insulation_score
    return insulation_scores


# def calculate_insulation_scores(contact_matrix, window_size):
#     insulation_scores_array = []
#     print(contact_matrix.shape[0])
#     print(window_size)
#     for i in range(contact_matrix.shape[0]):
#         if i < window_size or i > contact_matrix.shape[0] - window_size:
#             continue
#         submatrix = contact_matrix[i - window_size:i, i:i + window_size]
#         mean_score = submatrix.mean()
#         insulation_scores_array.append(mean_score * 100)
#
#     return normalize_insulation_scores(np.array(insulation_scores_array))


def calculate_insulation_scores(contact_matrix, window_size):
    insulation_scores_array = []
    matrix_size = contact_matrix.shape[0]

    # Calculate insulation scores
    for i in range(matrix_size):
        if i < window_size or i > matrix_size - window_size:
            insulation_scores_array.append(0)
            continue
        submatrix = contact_matrix[i - window_size:i, i:i + window_size]
        mean_score = submatrix.mean()
        insulation_scores_array.append(mean_score * 100)

    # Pad insulation scores array with zeros
    padding_length = matrix_size - len(insulation_scores_array)
    insulation_scores_array = np.pad(insulation_scores_array,
                                     (padding_length, 0), mode='constant')

    return normalize_insulation_scores(np.array(insulation_scores_array))


# def plot_scores_states(scores, states):
#     # Define colors for states
#
#     states = reorganize_states(states)
#
#     state_colors = ['purple' if state == 1 else 'green' if state == 0 else
#     'red' for state in states]
#     # Define colors for positive and negative values
#     score_colors = ['red' if score >= 0 else 'green' for score in scores]
#
#     # Create subplots
#     fig, (ax1, ax2) = plt.subplots(2, 1)
#
#     # Plot scores in the first subplot
#     ax1.bar(np.arange(len(scores)), scores, color=score_colors)
#
#     # Plot states in the second subplot
#     ax2.scatter(np.arange(len(states)), states, c=state_colors)
#
#     # Set titles and labels
#     ax1.set_title('Scores Results')
#     ax1.set_xlabel('HMM States')
#     ax1.set_ylabel('Sum of Scores')
#
#     ax2.set_xlabel('Bin #')
#     ax2.set_ylabel('State')
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()


def calculate_weighted_insulation_score(matrix, bin_size, square_size):
    # Calculate the dimensions of the matrix
    matrix_size = len(matrix)

    # Calculate the number of bins
    num_bins = matrix_size // bin_size

    # Calculate the number of bins in the square
    num_bins_in_square = square_size // bin_size

    # Initialize a list to store insulation scores
    insulation_scores = []

    # Iterate over each bin
    for i in range(num_bins):
        # Skip bins within 500kb of matrix start/end
        if i < num_bins_in_square // 2 or i > num_bins - num_bins_in_square // 2:
            continue

        # Calculate the indices of the square around the current bin
        start_index = max(0, i - num_bins_in_square // 2)
        end_index = min(num_bins, i + num_bins_in_square // 2 + 1)

        # Initialize a variable to store the total weighted signal within the square
        total_weighted_signal = 0
        total_elements = 0  # Count of elements to average

        # Iterate over each row in the square
        for j in range(start_index, end_index):
            # Calculate the indices for the square section in this row
            row_start_col = max(0, i - num_bins_in_square // 2)
            row_end_col = min(num_bins, i + num_bins_in_square // 2 + 1)

            # Iterate over each column within the square
            for k in range(row_start_col, row_end_col):
                # Apply different weights based on the position relative to the diagonal
                weight = 10000000 if k >= j else 0.000004
                total_weighted_signal += matrix[j, k] * weight
                total_elements += 1

        # Calculate the mean weighted signal within the square
        if total_elements > 0:
            mean_weighted_signal = total_weighted_signal
        else:
            mean_weighted_signal = 0

        # Assign the mean weighted signal to the current bin
        insulation_scores.append(mean_weighted_signal)

    # Normalize insulation scores
    mean_insulation = np.mean(insulation_scores)
    normalized_insulation_scores = [np.log2(insulation_score / mean_insulation)
                                    if mean_insulation > 0 else 0
                                    # Avoid log2(0)
                                    for insulation_score in
                                    insulation_scores]

    return insulation_scores


def calculate_gradual_weighted_insulation_score(matrix, bin_size, square_size):
    # Calculate the dimensions of the matrix
    matrix_size = len(matrix)

    # Calculate the number of bins
    num_bins = matrix_size // bin_size

    # Calculate the number of bins in the square
    num_bins_in_square = square_size // bin_size

    # Initialize a list to store insulation scores
    insulation_scores = []

    # Define the maximum and minimum weights
    max_weight = 10
    min_weight = 0.0005

    # Iterate over each bin
    for i in range(num_bins):
        # Skip bins within 500kb of matrix start/end
        if i < num_bins_in_square // 2 or i > num_bins - num_bins_in_square // 2:
            continue

        # Initialize variables for the weighted signal and element count
        total_weighted_signal = 0
        total_elements = 0

        # Determine the square indices
        start_index = max(0, i - num_bins_in_square // 2)
        end_index = min(num_bins, i + num_bins_in_square // 2 + 1)

        # Iterate over each row in the square
        for j in range(start_index, end_index):
            # Calculate the indices for the current row section
            row_start_col = max(0, i - num_bins_in_square // 2)
            row_end_col = min(num_bins, i + num_bins_in_square // 2 + 1)

            # Iterate over each column within the square section
            for k in range(row_start_col, row_end_col):
                # Determine the distance from the diagonal
                distance_from_diagonal = k - j

                # Adjust the weight based on the distance from the diagonal
                if distance_from_diagonal >= 0:
                    # Above or on the diagonal
                    weight = max_weight - ((
                                                   max_weight - min_weight) / num_bins_in_square) * distance_from_diagonal
                else:
                    # Below the diagonal
                    weight = min_weight - ((
                                                   min_weight - max_weight) / num_bins_in_square) * abs(
                        distance_from_diagonal)

                # Ensure weights do not exceed boundaries
                weight = max(min(weight, max_weight), min_weight)

                # Accumulate the weighted signal and count the elements
                total_weighted_signal += matrix[j, k] * weight
                total_elements += 1

        # Calculate the mean weighted signal within the square
        mean_weighted_signal = total_weighted_signal / total_elements if total_elements > 0 else 0

        # Append the calculated mean weighted signal to the insulation scores
        insulation_scores.append(mean_weighted_signal)

    # Normalize insulation scores
    mean_insulation = np.mean(insulation_scores)
    normalized_insulation_scores = [
        np.log2(score / mean_insulation) if mean_insulation > 0 else 0 for
        score in insulation_scores]

    return insulation_scores


def calculate_gradual_weighted_insulation_score(matrix, bin_size, square_size):
    # Calculate the dimensions of the matrix
    matrix_size = len(matrix)

    # Calculate the number of bins
    num_bins = matrix_size // bin_size

    # Calculate the number of bins in the square
    num_bins_in_square = square_size // bin_size

    # Initialize a list to store insulation scores
    insulation_scores = []

    # Define the maximum and minimum weights
    max_weight = 10
    min_weight = 0.0005

    # Iterate over each bin
    for i in range(num_bins):
        # Skip bins within 500kb of matrix start/end
        if i < num_bins_in_square // 2 or i > num_bins - num_bins_in_square // 2:
            continue

        # Initialize variables for the weighted signal and element count
        total_weighted_signal = 0
        total_elements = 0

        # Determine the square indices
        start_index = max(0, i - num_bins_in_square // 2)
        end_index = min(num_bins, i + num_bins_in_square // 2 + 1)

        # Iterate over each row in the square
        for j in range(start_index, end_index):
            # Calculate the indices for the current row section
            row_start_col = max(0, i - num_bins_in_square // 2)
            row_end_col = min(num_bins, i + num_bins_in_square // 2 + 1)

            # Iterate over each column within the square section
            for k in range(row_start_col, row_end_col):
                # Determine the distance from the diagonal
                distance_from_diagonal = k - j

                # Adjust the weight based on the distance from the diagonal
                if distance_from_diagonal >= 0:
                    # Above or on the diagonal
                    weight = max_weight - ((
                                                   max_weight - min_weight) / num_bins_in_square) * distance_from_diagonal
                else:
                    # Below the diagonal
                    weight = min_weight - ((
                                                   min_weight - max_weight) / num_bins_in_square) * abs(
                        distance_from_diagonal)

                # Ensure weights do not exceed boundaries
                weight = max(min(weight, max_weight), min_weight)

                # Accumulate the weighted signal and count the elements
                total_weighted_signal += matrix[j, k] * weight
                total_elements += 1

        # Calculate the mean weighted signal within the square
        mean_weighted_signal = total_weighted_signal / total_elements if total_elements > 0 else 0

        # Append the calculated mean weighted signal to the insulation scores
        insulation_scores.append(mean_weighted_signal)

    # Normalize insulation scores
    mean_insulation = np.mean(insulation_scores)
    normalized_insulation_scores = [
        np.log2(score / mean_insulation) if mean_insulation > 0 else 0 for
        score in insulation_scores]

    return insulation_scores


# def plot_scores_states_boundaries(scores, states, boundaries, hic_matrix,
#                                   start_idx, end_idx, title_name=''):
#     # Define colors for states
#     # states = reorganize_states(states)
#     state_colors = [
#         'red' if state == 1 else 'green' if state == 2 else 'blue' for state
#         in states
#     ]
#
#     # Define colors for positive and negative values
#     score_colors = ['red' if score >= 0 else 'green' for score in scores]
#
#     # Create subplots
#     # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))
#
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16),
#                                              gridspec_kw={
#                                                  'height_ratios': [2, 1, 1, 1],
#                                                  'hspace': 0.5})
#
#     # Plot contact matrix pyramid
#     upper_triangular = np.triu(hic_matrix, k=1)
#     i, j = np.where(upper_triangular)
#     r2 = upper_triangular[i, j]
#     y = (j - i) / 2
#     df = pd.DataFrame({'i': i + y, 'j': y, 'r2': r2})
#     norm = mcolors.Normalize(vmin=df['r2'].min(), vmax=df['r2'].max())
#     im = ax1.scatter(df['i'], df['j'], c=df['r2'], cmap='Reds', norm=norm,
#                      alpha=1.0, s=25, edgecolors='none')
#     # cbar = fig.colorbar(im, ax=ax1, label='Normalized Interacting Counts',
#     #                     orientation='horizontal',
#     #                     pad=0.1)  # Position colorbar horizontally above ax1
#     # cbar.ax.xaxis.set_ticks_position('bottom')  # Move colorbar ticks to top
#
    # ax1.set_aspect(
    #     'auto')  # Let's make the aspect ratio automatic for the first subplot
#     ax1.set_yticks([])
#     ax1.set_ylabel(None)
#     ax1.set_title('Hi-C Matrix')
#
#     # Set the x-axis limits for the third plot to match the first two plots
#     # ax1.set_xlim([start_idx, end_idx - 1])
#     # ax1.set_xticks(range(start_idx, end_idx), 25)
#     ax1.set_xlabel('Chr [Nucleotides]')
#     ax1.set_title('Hi-C Matrix')
#     ax1.grid()
#
#     # Plot scores in the first subplot
#     ax2.bar(np.arange(start_idx, end_idx), scores,
#             color=score_colors)
#
#     # Plot states in the second subplot
#     ax3.scatter(np.arange(start_idx, end_idx), states,
#                 color=state_colors)
#     ax3.set_title('Gaussian HMM State - Emissions')
#
#     # Plot boundaries in the third subplot
#     ax4.axhline(y=0, color='r', linestyle='-')
#     # Transform boundaries to be within the range specified by start_idx and end_idx
#     # transformed_boundaries = np.array(
#     #     [b for b in boundaries if start_idx <= b < end_idx])
#     ax4.scatter(boundaries, np.zeros_like(
#         boundaries), c='b')
#
#     # Set titles and labels
#     ax2.set_title('Scores Results')
#     ax2.set_xlabel('Chr [Nucleotides]')
#     ax2.set_ylabel('Sum of Scores')
#
#     ax3.set_xlabel('Chromosomal Position [bp]')
#     ax3.set_ylabel('State')
#
#     ax4.set_title('TADs Boundaries Estimation')
#     ax4.set_xlabel('Chr [Nucleotides]')
#     ax4.set_ylabel('Boundary')
#
#     # Set the x-axis limits for the third plot to match the first two plots
#     # ax4.set_xlim([start_idx, end_idx - 1])
#     # ax4.set_xticks(range(start_idx, end_idx), 25)
#
#     # Set x-tick labels to display the desired values
#     ax1.set_xticklabels(
#         [(i + start_idx) * 40 * 1024 for i in ax3.get_xticks()])
#     ax2.set_xticklabels(
#         [(i + start_idx) * 40 * 1024 for i in ax2.get_xticks()])
#     ax3.set_xticklabels(
#         [(i + start_idx) * 40 * 1024 for i in ax3.get_xticks()])
#     ax4.set_xticklabels(
#         [(i + start_idx) * 40 * 1024 for i in ax4.get_xticks()])
#
#     ax1.grid()
#     ax2.grid()
#     ax3.grid()
#     ax4.grid()
#     fig.suptitle(f'Division to TADs HMM Estimation - {title_name}',
#                  fontsize=16)  # Add title to the whole figure
#     # Show the plot
#     plt.tight_layout()
#     plt.show()


def plot_scores_states_boundaries(scores, states, boundaries, hic_matrix,
                                  start_idx, end_idx, title_name=''):
    # Define colors for states
    state_colors = ['red' if state == 1 else 'green' if state == 2 else 'blue' for state in states]

    # Define colors for positive and negative values
    score_colors = ['red' if score >= 0 else 'green' for score in scores]

    # Create subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16),
                                             gridspec_kw={'height_ratios': [2, 1, 1, 1], 'hspace': 0.5})

    # Set x-ticks and x-tick labels
    tick_positions = np.arange(start_idx, end_idx, 25)  # Adjust the step size as needed
    tick_labels = [(i * 40 * 1000) for i in tick_positions]

    # Plot contact matrix pyramid
    upper_triangular = np.triu(hic_matrix, k=1)
    i, j = np.where(upper_triangular)
    r2 = upper_triangular[i, j]
    y = (j - i) / 2
    df = pd.DataFrame({'i': i + y, 'j': y, 'r2': r2})
    norm = mcolors.Normalize(vmin=df['r2'].min(), vmax=df['r2'].max())
    im = ax1.scatter(df['i'], df['j'], c=df['r2'], cmap='Reds', norm=norm,
                     alpha=1.0, s=25, edgecolors='none')
    # cbar = fig.colorbar(im, ax=ax1, label='Normalized Interacting Counts',
    #                     orientation='horizontal',
    #                     pad=0.1)  # Position colorbar horizontally above ax1
    # cbar.ax.xaxis.set_ticks_position('bottom')  # Move colorbar ticks to top

    ax1.set_aspect(
        'auto')  # Let's make the aspect ratio automatic for the first subplot
    ax1.set_yticks([])
    ax1.set_ylabel(None)
    ax1.set_title('Hi-C Matrix')

    # Set the x-axis limits for the third plot to match the first two plots
    # ax1.set_xlim([start_idx, end_idx - 1])
    # ax1.set_xticks(range(start_idx, end_idx), 25)
    ax1.set_xlabel('Chromosomal Position [bp]')
    ax1.set_title('Hi-C Matrix')
    ax1.grid()

    # Plot scores in the first subplot
    ax2.bar(np.arange(start_idx, end_idx), scores,
            color=score_colors)

    # Plot states in the second subplot
    ax3.scatter(np.arange(start_idx, end_idx), states,
                color=state_colors)
    ax3.set_title('Gaussian HMM State - Emissions')

    # Plot boundaries in the fourth subplot
    ax4.axhline(y=0, color='r', linestyle='-')
    print(title_name)
    for i in range(len(np.nonzero(boundaries)[0]) - 1):
        lim_start = (start_idx + np.nonzero(boundaries)[0][i]) * 1000 * 40
        lim_end = (start_idx + np.nonzero(boundaries)[0][i + 1]) * 1000 * 40
        print(f'chr15    {lim_start}    {lim_end}')
    print(np.nonzero(boundaries))
    boundary_indices = np.nonzero(boundaries)[0] + start_idx
    ax4.scatter(boundary_indices, np.zeros_like(boundary_indices), c='b', label='TAD Boundary')

    # Set the x-axis limits and ticks for all subplots
    for ax in [ax2, ax3, ax4]:
        ax.set_xlim([start_idx - 10, end_idx + 10])
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.grid()

    # ax1.set_xlim([start_idx, end_idx])
    ax1.set_xticklabels(
                [(i + start_idx) * 40 * 1000 for i in ax1.get_xticks()])

    # Set titles and labels
    ax2.set_title('Scores Results')
    ax2.set_xlabel('Chromosomal Position [bp]')
    ax2.set_ylabel('Sum of Scores')

    ax3.set_xlabel('Chromosomal Position [bp]')
    ax3.set_ylabel('State')

    ax4.set_title('TADs Boundaries Estimation')
    ax4.set_xlabel('Chromosomal Position [bp]')
    ax4.set_ylabel('Boundary')

    # Set the supertitle for the entire figure
    fig.suptitle(f'Division to TADs HMM Estimation - {title_name}', fontsize=16)

    # Show the plot
    plt.tight_layout()
    plt.show()




# def plot_scores_states_boundaries(scores, states, boundaries, hic_matrix,
#                                   start_idx, end_idx, title_name=''):
#     # Define colors for states based on your defined criteria
#     state_colors = ['red' if state == 1 else 'green' if state == 2 else 'blue'
#                     for state in states]
#
#     # Define colors for positive and negative values
#     score_colors = ['red' if score >= 0 else 'green' for score in scores]
#
#     # Create subplots with shared x-axes
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16),
#                                              sharex=True,
#                                              gridspec_kw={
#                                                  'height_ratios': [2, 1, 1, 1],
#                                                  'hspace': 0.5})
#
#     # Set the number of nucleotides per bin
#     nucleotides_per_bin = 40 * 1024  # Assuming 40 kb per bin
#
#     # Adjust the tick positions and labels
#     tick_positions = np.arange(start_idx, end_idx, 25)  # Ticks every 25 positions
#     tick_labels = [(pos * nucleotides_per_bin) for pos in tick_positions]
#
#     # Plot the Hi-C contact matrix pyramid
#     # The Hi-C matrix needs to be sliced according to start_idx and end_idx before plotting
#     hi_c_matrix_slice = hic_matrix[start_idx:end_idx, start_idx:end_idx]
#     upper_triangular = np.triu(hi_c_matrix_slice, k=1)
#     i, j = np.where(upper_triangular)
#     r2 = upper_triangular[i, j]
#     y = (j - i) / 2
#     df = pd.DataFrame({'i': i + y, 'j': y, 'r2': r2})
#     norm = mcolors.Normalize(vmin=df['r2'].min(), vmax=df['r2'].max())
#     ax1.scatter(i + y, y, c=r2, cmap='Reds', norm=norm, alpha=1.0, s=25, edgecolors='none')
#     ax1.set_aspect('equal')
#     ax1.set_title('Hi-C Matrix')
#
#     # Plot scores in the second subplot
#     ax2.bar(tick_positions, scores, color=score_colors)  # Slice scores every 25 positions
#     ax2.set_title('Scores Results')
#
#     # Plot states in the third subplot
#     ax3.scatter(tick_positions, states, color=state_colors)  # Slice states every 25 positions
#     ax3.set_title('Gaussian HMM State - Emissions')
#
#     # Plot boundaries in the fourth subplot
#     ax4.axhline(y=0, color='r', linestyle='-')
#     transformed_boundaries = np.array(
#         [b - start_idx for b in boundaries if start_idx <= b < end_idx])
#     ax4.scatter(transformed_boundaries, np.zeros_like(transformed_boundaries), color='b')
#     ax4.set_title('TADs Boundaries Estimation')
#
#     # Apply the tick positions and labels to all subplots
#     for ax in [ax1, ax2, ax3, ax4]:
#         ax.set_xticks(tick_positions)
#         ax.set_xticklabels(tick_labels, rotation=90)
#         ax.grid(True)
#
#     # Set axis labels
#     ax1.set_ylabel('Hi-C Interactions')
#     ax2.set_ylabel('Score')
#     ax3.set_ylabel('State')
#     ax4.set_xlabel('Chromosomal Position (bp)')
#     ax4.set_ylabel('Boundary')
#
#     # Set the supertitle for the entire figure
#     fig.suptitle(f'Division to TADs HMM Estimation - {title_name}', fontsize=16)
#
#     # Show the plot
#     plt.tight_layout()
#     plt.show()


def reorganize_states(states):
    states = [2 if state == 0 else 0 if state == 1 else 1 if state == 0
    else state for \
              state in
              states]
    return states


# get mtx matrix and calculate for all diag the expected.
def ave(matrix):
    n = len(matrix)  # Assuming it's a square matrix.
    diagonal_sums = [0] * n

    # Calculate sum of each diagonal
    for col_start in range(n):
        for row, col in zip(range(n), range(col_start, n)):
            diagonal_sums[col_start] += matrix[row][col]

    # Update the matrix based on diagonal sums
    for col_start in range(n):
        for row, col in zip(range(n), range(col_start, n)):
            if (diagonal_sums[col_start] != 0):
                matrix[row][col] /= diagonal_sums[col_start]

    return matrix


def pca_A_B_compartment(hic_matrix, title_name='Cortex'):
    fig, ax1 = plt.subplots(
        figsize=(8, 6))  # Create a subplot with a single axis
    data_size = hic_matrix.shape[0]
    bin_size = 40
    pca_score = np.zeros(data_size)
    resolution = int(data_size / bin_size)

    for i in range(0, data_size, resolution):
        submatrix = hic_matrix[i:i + resolution, i:i + resolution]
        t = ave(submatrix)
        pca = PCA(n_components=1)
        pca.fit(t)
        pca_score[i:i + resolution] = pca.components_[0]
    positive_values = np.where(pca_score <= 0, pca_score, 0)
    non_positive_values = np.where(pca_score > 0, pca_score, 0)

    # Plotting
    ax1.bar(range(len(positive_values)), positive_values, color='green',
            label='A')
    ax1.bar(range(len(non_positive_values)), non_positive_values,
            color='red', label='B')
    ax1.legend()
    fig.suptitle(f'A-B Compartment - {title_name}',
                 fontsize=16)  # Add title to the whole figure
    ax1.set_xlabel('Chr Interval [Nucleotides]')
    ax1.set_ylabel('Value')
    ax1.set_xticklabels(
        [(i + GENOME_START_IDX) * 40 * 1024 for i in ax1.get_xticks()])
    ax1.set_title(f"Resolution - Bin Size={bin_size}")
    fig.show()


if __name__ == '__main__':
    # file_path = "CO\\outputs\\CO.chr2.prob.tad.matrix.txt"
    file_path_co = "CO\\CO.HiCNorm.chr2.txt"
    file_path_hc = "HC\\HC.HiCNorm.chr2.txt"
    file_path_li = "LI\\LI.nor.chr2.txt"
    file_path_ao = "AO\\AO.nor.chr3.txt"
    file_path_pa = "PA\\PA."
    file_path_lv13 = "LV\\LV.nor.chr13.txt"
    file_path_rv13 = "RV\\RV.nor.chr13.txt"
    file_path_lv14 = "LV\\LV.nor.chr14.txt"
    file_path_rv14 = "RV\\RV.nor.chr14.txt"
    file_path_lv21 = "LV\\LV.nor.chr21.txt"
    file_path_rv21 = "RV\\RV.nor.chr21.txt"


    file_path_tro = "tro\\tro.rep1.nor.chr2.txt"
    file_path_mes = "mes\\mes.rep1.nor.chr2.txt"
    file_path_mes15 = "mes\\mes.rep1.nor.chr15.txt"
    file_path_h1 = "H1\\h1.rep2.nor.chr15.txt"

    file_path_cortex = "CO\\outputs\\CO.chr2.matrix.txt"
    file_path_hippocampus = "HC\\outputs\\HC.chr2.matrix.txt"

    hic_matrix_co = load_hic_matrix(file_path_co)
    hic_matrix_hc = load_hic_matrix(file_path_hc)
    hic_matrix_li = load_hic_matrix(file_path_li)

    hic_matrix_tro = load_hic_matrix(file_path_tro)
    hic_matrix_mes = load_hic_matrix(file_path_mes)
    hic_matrix_mes15 = load_hic_matrix(file_path_mes15)
    hic_matrix_h1 = load_hic_matrix(file_path_h1)

    # contact_matrix_lv13 = load_hic_matrix(file_path_lv13)
    # contact_matrix_rv13 = load_hic_matrix(file_path_rv13)
    # contact_matrix_lv14 = load_hic_matrix(file_path_lv14)
    # contact_matrix_rv14 = load_hic_matrix(file_path_rv14)
    # contact_matrix_lv21 = load_hic_matrix(file_path_lv21)
    # contact_matrix_rv21 = load_hic_matrix(file_path_rv21)

    # plot_hic_pyramids(hic_matrix_co[0:100, 0:100], start_idx=0)
    # plot_hic_pyramids(hic_matrix_hc[0:100, 0:100], start_idx=0)
    # plot_hic_pyramids(hic_matrix_li[0:100, 0:100], start_idx=0)
    #
    # plot_hic_pyramids(hic_matrix_co[3000:3100, 3000:3100], start_idx=3000)
    # plot_hic_pyramids(hic_matrix_hc[3000:3100, 3000:3100], start_idx=3000)
    # plot_hic_pyramids(hic_matrix_li[3000:3100, 3000:3100], start_idx=3000)

    # plot_hic_pyramids(contact_matrix_co[0:100, 0:100], start_idx=0)
    # plot_hic_pyramids(contact_matrix_hc[0:100, 0:100],start_idx=0)

    # Define bin size and square size in kb
    bin_size_kb = 40
    square_size_kb = 400

    # Convert bin size and square size to number of bins
    bin_size = bin_size_kb // 40
    square_size = square_size_kb // 40

    # compartment = run_pca(hic_matrix)
    # plot_compartment_vector(compartment)

    insu_scores_co = calculate_insulation_scores(hic_matrix_co, window_size=25)
    insu_scores_hc = calculate_insulation_scores(hic_matrix_hc, window_size=25)
    insu_scores_li = calculate_insulation_scores(hic_matrix_li, window_size=25)
    insu_scores_tro = calculate_insulation_scores(hic_matrix_tro,
                                                  window_size=25)
    insu_scores_mes = calculate_insulation_scores(hic_matrix_mes,
                                                  window_size=25)
    insu_scores_mes15 = calculate_insulation_scores(hic_matrix_mes15,
                                                    window_size=25)
    insu_scores_h1 = calculate_insulation_scores(hic_matrix_h1,
                                                 window_size=25)
    #
    #
    # print(f"Insulation Scores CO Length: {len(insu_scores_co)}")
    # print(f"Insulation Scores CO Length: {len(insu_scores_hc)}")
    #
    # # Calculate insulation scores
    # # insulation_scores = calculate_insulation_score(hic_matrix, bin_size,
    # #                                                square_size)
    #
    # # insulation_scores_weighted = calculate_weighted_insulation_score(
    # #     hic_matrix, bin_size, square_size)
    # # #
    # # insulation_scores_gradual = calculate_gradual_weighted_insulation_score(
    # #     hic_matrix, bin_size, square_size)
    #
    predicted_states, hmm_model, scores, means, covs, transition_points = \
        train_hmm_model(
            np.array(
                hic_matrix_co), compute_score, bin_size, square_size,
            score_method_str='di_score')
    #
    predicted_states_hc, hmm_model_hc, scores_hc, means_hc, covs_hc, \
    transition_points_hc = \
        train_hmm_model(
            np.array(
                hic_matrix_hc), compute_score, bin_size, square_size,
            score_method_str='di_score')
    #
    predicted_states_li, hmm_model_li, scores_li, means_li, covs_li, \
    transition_points_li = \
        train_hmm_model(
            np.array(
                hic_matrix_li), compute_score, bin_size, square_size,
            score_method_str='di_score')

    predicted_states_tro, hmm_model_tro, scores_tro, means_tro, covs_tro, \
    transition_points_tro = \
        train_hmm_model(
            np.array(
                hic_matrix_tro), compute_score, bin_size, square_size,
            score_method_str='di_score')

    predicted_states_mes, hmm_model_mes, scores_mes, means_mes, covs_mes, \
    transition_points_mes = \
        train_hmm_model(
            np.array(
                hic_matrix_mes), compute_score, bin_size, square_size,
            score_method_str='di_score')

    predicted_states_mes15, hmm_model_mes15, scores_mes15, means_mes15, \
    covs_mes15, \
    transition_points_mes15 = \
        train_hmm_model(
            np.array(
                hic_matrix_mes15), compute_score, bin_size, square_size,
            score_method_str='di_score')

    predicted_states_h1, hmm_model_h1, scores_h1, means_h1, covs_h1, \
    transition_points_h1 = \
        train_hmm_model(
            np.array(
                hic_matrix_h1), compute_score, bin_size, square_size,
            score_method_str='di_score')

    # predicted_states_lv13, hmm_model_lv13, scores_lv13, means_lv13, covs_lv13, \
    # transition_points_lv13 = \
    #     train_hmm_model(
    #         np.array(
    #             contact_matrix_lv13), compute_score, bin_size, square_size,
    #         score_method_str='di_score')
    #
    # predicted_states_rv13, hmm_model_rv13, scores_rv13, means_rv13, \
    # covs_rv13, \
    # transition_points_rv13 = \
    #     train_hmm_model(
    #         np.array(
    #             contact_matrix_rv13), compute_score, bin_size, square_size,
    #         score_method_str='di_score')
    #
    # predicted_states_lv14, hmm_model_lv14, scores_lv14, means_lv14, \
    # covs_lv14, \
    # transition_points_lv14 = \
    #     train_hmm_model(
    #         np.array(
    #             contact_matrix_lv14), compute_score, bin_size, square_size,
    #         score_method_str='di_score')
    #
    # predicted_states_rv14, hmm_model_rv14, scores_rv14, means_rv14, \
    # covs_rv14, \
    # transition_points_rv14 = \
    #     train_hmm_model(
    #         np.array(
    #             contact_matrix_rv14), compute_score, bin_size, square_size,
    #         score_method_str='di_score')
    #
    #
    # predicted_states_lv21, hmm_model_lv21, scores_lv21, means_lv21, \
    # covs_lv21, \
    # transition_points_lv21 = \
    #     train_hmm_model(
    #         np.array(
    #             contact_matrix_lv21), compute_score, bin_size, square_size,
    #         score_method_str='di_score')
    #
    # predicted_states_rv21, hmm_model_rv21, scores_rv21, means_rv21, \
    # covs_rv21, \
    # transition_points_rv21 = \
    #     train_hmm_model(
    #         np.array(
    #             contact_matrix_rv21), compute_score, bin_size, square_size,
    #         score_method_str='di_score')

    # plot_hmm_model_distribution(hmm_model)
    # plot_hmm_model_distribution(hmm_model_hc)
    # plot_hmm_model_distribution(hmm_model_li)

    emission_mes15 = calculate_emission_probabilities(predicted_states_mes15)
    print(f'Mesendoderm Ce'
                                     f'll Emission Pr'
                                     f'obabilities: {emission_mes15}')
    emission_h1 = calculate_emission_probabilities(predicted_states_h1)
    print(f'Embryonic Stem Cell Emission Pr'
                                     f'obabilities: {emission_h1}')
    emission_mes15_small = calculate_emission_probabilities(
        predicted_states_mes15[1500:2500])
    print(f'Mesendoderm Ce'
                                     f'll Emission Pr'
                                     f'obabilities 1500-2500:'
          f' {emission_mes15_small}')
    emission_h1_small = calculate_emission_probabilities(predicted_states_h1[
                                                     1500:2500])
    print(f'Embryonic Stem Cell Emission Pr'
                                     f'obabilities 1500-25'
          f'00: {emission_h1_small}')

    emission_tro = calculate_emission_probabilities(predicted_states_tro)
    print(f'Trophoblasts-Like Cell Emission Pr'
                                     f'obabilities: {emission_tro}')

    emission_mes = calculate_emission_probabilities(predicted_states_mes)
    print(f'Mesendoderm Ce'
                                     f'll Emission Pr'
                                     f'obabilities: {emission_mes}')

    emission_cortex = calculate_emission_probabilities(predicted_states)
    print(f'Cortex Emission Pr'
          f'obabilities: {emission_cortex}')

    emission_hc = calculate_emission_probabilities(predicted_states_hc)
    print(f'Hippocampus Emission Pr'
          f'obabilities: {emission_hc}')

    emission_li = calculate_emission_probabilities(predicted_states_li)
    print(f'Liver Emission Pr'
          f'obabilities: {emission_li}')


    plot_scores_pyramids_comparison_graphs(scores[4000:4100], insu_scores_co[
                                                              4000:4100],
                                           hic_matrix_co[4000:4100, 4000:4100],
                                           cell_name='CO Index 4000-4100',
                                           start_idx=4000)

    plot_scores_pyramids_comparison_graphs(scores_hc[4000:4100],
                                           insu_scores_hc[4000:4100],
                                           hic_matrix_hc[4000:4100, 4000:4100],
                                           cell_name='HC Index 4000-4100',
                                           start_idx=4000)

    plot_scores_pyramids_comparison_graphs(scores_li[4000:4100],
                                           insu_scores_li[
                                           4000:4100],
                                           hic_matrix_li[4000:4100, 4000:4100],
                                           cell_name='LI Index 4000-4100',
                                           start_idx=4000)

    plot_scores_pyramids_comparison_graphs(scores[3000:3100], insu_scores_co[
                                                              3000:3100],
                                           hic_matrix_co[3000:3100, 3000:3100],
                                           cell_name='CO Index 3000-3100',
                                           start_idx=3000)

    plot_scores_pyramids_comparison_graphs(scores_hc[3000:3100],
                                           insu_scores_hc[3000:3100],
                                           hic_matrix_hc[3000:3100, 3000:3100],
                                           cell_name='HC Index 3000-3100',
                                           start_idx=3000)

    plot_scores_pyramids_comparison_graphs(scores_li[3000:3100],
                                           insu_scores_li[3000:3100],
                                           hic_matrix_li[3000:3100, 3000:3100],
                                           cell_name='LI Index 3000-3100',
                                           start_idx=3000)

    plot_scores_pyramids_comparison_graphs(scores_tro[3000:3100],
                                           insu_scores_tro[3000:3100],
                                           hic_matrix_tro[3000:3100,
                                           3000:3100],
                                           cell_name='Trophoblasts-Like Cell'
                                                     ,
                                           start_idx=3000)

    plot_scores_pyramids_comparison_graphs(scores_mes[3000:3100],
                                           insu_scores_mes[3000:3100],
                                           hic_matrix_mes[3000:3100,
                                           3000:3100],
                                           cell_name='Mesendoderm Cell',
                                           start_idx=3000)

    plot_scores_pyramids_comparison_graphs(scores_mes15[1200:1300],
                                           insu_scores_mes15[1200:1300],
                                           hic_matrix_mes15[1200:1300,
                                           1200:1300],
                                           cell_name='Mesendoderm Cell CHR15 '
                                                     'Index '
                                                     '1200-1300',
                                           start_idx=3000)

    plot_scores_pyramids_comparison_graphs(scores_h1[1200:1300],
                                           insu_scores_h1[1200:1300],
                                           hic_matrix_h1[1200:1300,
                                           1200:1300],
                                           cell_name='Embryonic Stem Cell '
                                                     'CHR15 '
                                                     'Index '
                                                     '1200-1300',
                                           start_idx=1200)

    # plot_scores_comparison_graphs(scores[:100], insu_scores_co[:100],
    #                               cell_name='DI VS. Insulation Scores - CO',
    #                               start_idx=0, end_idx=100)
    # plot_scores_comparison_graphs(scores_hc[:100], insu_scores_hc[:100],
    #                               cell_name='DI VS. Insulation Scores - HC',
    #                               start_idx=0, end_idx=100)
    # plot_scores_comparison_graphs(scores_li[:100], insu_scores_li[:100],
    #                               cell_name='DI VS. Insulation Scores - HC',
    #                               start_idx=0, end_idx=100)

    plot_scores_comparison_graphs(scores[3000:3100], insu_scores_co[3000:3100],
                                  cell_name='CO Index 3000-3100',
                                  start_idx=3000, end_idx=3100)
    plot_scores_comparison_graphs(scores_tro[3000:3100], insu_scores_tro[
                                                         3000:3100],
                                  cell_name='tro Index 3000-3100',
                                  start_idx=3000, end_idx=3100)
    plot_scores_comparison_graphs(scores_mes[3000:3100], insu_scores_mes[
                                                         3000:3100],
                                  cell_name='mes Index 3000-3100',
                                  start_idx=3000, end_idx=3100)
    # plot_scores_comparison_graphs(scores[4000:4100], insu_scores_co[4000:4100],
    #                               cell_name='CO Index 4000-4200',
    #                               start_idx=4000, end_idx=4100)
    # plot_scores_comparison_graphs(scores[300:400], insu_scores_co[300:400],
    #                               cell_name='CO Index 300-400', start_idx=300,
    #                               end_idx=400)
    # plot_scores_comparison_graphs(scores[400:500], insu_scores_co[400:500],
    #                               cell_name='CO Index 400-500', start_idx=400,
    #                               end_idx=500)

    # boundaries = predict_tads(predicted_states[:200])
    boundaries_co = predict_tads_with_flags(predicted_states,
                                            start_idx=0, end_idx=6080)
    boundaries_hc = predict_tads_with_flags(predicted_states_hc,
                                 start_idx=0, end_idx=6080)
    boundaries_li = predict_tads_with_flags(predicted_states_li,
                                 start_idx=0, end_idx=6080)
    boundaries_tro = predict_tads_with_flags(predicted_states_tro[2000:2200],
                                  start_idx=2000, end_idx=2200)
    boundaries_mes = predict_tads_with_flags(predicted_states_mes[2000:2200],
                                  start_idx=2000, end_idx=2200)
    boundaries_h1 = predict_tads_with_flags(predicted_states_h1,
                                 start_idx=0, end_idx=2564)
    boundaries_mes15 = predict_tads_with_flags(predicted_states_mes15,
                                    start_idx=0, end_idx=2564)
    # boundaries_lv13 = predict_tads_with_flags(predicted_states_lv13,
    #                              start_idx=0, end_idx=6080)
    # boundaries_rv13 = predict_tads_with_flags(predicted_states_rv13,
    #                              start_idx=0, end_idx=6080)
    # boundaries_lv14 = predict_tads_with_flags(predicted_states_lv14,
    #                              start_idx=0, end_idx=6080)
    # boundaries_rv14 = predict_tads_with_flags(predicted_states_rv14,
    #                              start_idx=0, end_idx=6080)
    # boundaries_lv21 = predict_tads_with_flags(predicted_states_lv21,
    #                              start_idx=0, end_idx=6080)
    # boundaries_rv21 = predict_tads_with_flags(predicted_states_rv21,
    #                              start_idx=0, end_idx=6080)

    plot_scores_states_boundaries(scores[3000:3200], predicted_states[
                                                     3000:3200],
                                  boundaries_co[3000:3200], hic_matrix_co[3000:3200,
                                                 3000:3200],
                                  3000,
                                  3200, title_name='Cortex Chr2 3000-3200')
    plot_scores_states_boundaries(scores_hc[3000:3200], predicted_states_hc[
                                                     3000:3200],
                                  boundaries_hc[3000:3200],
                                  hic_matrix_hc[3000:3200,
                                  3000:3200],
                                  3000,
                                  3200, title_name='Hippocampus Chr2 '
                                                   '3000-3200')
    plot_scores_states_boundaries(scores_li[3000:3200], predicted_states_li[
                                                     3000:3200],
                                  boundaries_li[3000:3200],
                                  hic_matrix_li[3000:3200,
                                  3000:3200],
                                  3000,
                                  3200, title_name='Liver Chr2 '
                                                   '3000-3200')
    plot_scores_states_boundaries(scores[4000:4200], predicted_states[
                                                        4000:4200],
                                  boundaries_co[4000:4200],
                                  hic_matrix_co[4000:4200, 4000:4200],
                                  4000,
                                  4200, title_name='Cortex Chr2 '
                                                   '4000-4200')
    plot_scores_states_boundaries(scores_hc[4000:4200], predicted_states_hc[
                                                        4000:4200],
                                  boundaries_hc,
                                  hic_matrix_hc[4000:4200, 4000:4200],
                                  4000,
                                  4200, title_name='Hippocampus Chr2 '
                                                   '4000-4200')
    plot_scores_states_boundaries(scores_li[4000:4200], predicted_states_li[
                                                        4000:4200],
                                  boundaries_li[4000:4200],
                                  hic_matrix_co[4000:4200, 4000:4200],
                                  4000,
                                  4200, title_name='Liver Chr2 '
                                                   '4000-4200')
    plot_scores_states_boundaries(scores[5000:5200], predicted_states[
                                                        5000:5200],
                                  boundaries_co[5000:5200],
                                  hic_matrix_co[5000:5200, 5000:5200
                                  ], 5000,
                                  5200, title_name='Cortex Chr2 5000-5200')
    plot_scores_states_boundaries(scores_hc[5000:5200], predicted_states_hc[
                                                        5000:5200],
                                  boundaries_hc[5000:5200],
                                  hic_matrix_hc[5000:5200, 5000:5200
                                  ], 5000,
                                  5200, title_name='Hippocampus Chr2 '
                                                   '5000-5200')
    plot_scores_states_boundaries(scores_li[5000:5200], predicted_states_li[
                                                        5000:5200],
                                  boundaries_li[5000:5200],
                                  hic_matrix_li[5000:5200, 5000:5200
                                  ], 5000,
                                  5200, title_name='Liver Chr2 5000-5200')
    # plot_scores_states_boundaries(scores_tro[2000:2200], predicted_states_tro[
    #                                                      2000:2200],
    #                               boundaries_tro,  hic_matrix_tro[2000:2200,
    #                                                2000:2200],
    #                               2000,
    #                               2200, title_name='Trophoblasts-Like Cell '
    #                                                'Chr2 2000-2200')
    # plot_scores_states_boundaries(scores_mes[2000:2200], predicted_states_mes[
    #                                                      2000:2200],
    #                               boundaries_mes, hic_matrix_mes[2000:2200,
    #                                               2000:2200],
    #                               2000,
    #                               2200, title_name='Mesendoderm Cell Chr2 '
    #                                                '2000-2200')
    # plot_scores_states_boundaries(scores_h1[2200:2400], predicted_states_h1[
    #                                                  2200:2400],
    #                               boundaries_h1[2200:2400], hic_matrix_h1[2200:2400,
    #                                              2200:2400],
    #                               2200,
    #                               2400, title_name='Embryonic Stem Cell ' \
    #                                                 'Chr15 '
    #                                                '2200-2400')
    # plot_scores_states_boundaries(scores_mes15[2200:2400],
    #                               predicted_states_mes15[
    #                               2200:2400],
    #                               boundaries_mes15[2200:2400], hic_matrix_mes15[
    #                                                  2200:2400,
    #                                                 2200:2400],
    #                               2200,
    #                               2400, title_name='Mesendoderm Cell Chr15 '
    #                                                '2200-2400')
    # plot_scores_states_boundaries(scores_h1[1000:1200], predicted_states_h1[
    #                                                     1000:1200],
    #                               boundaries_h1[1000:1200], hic_matrix_h1[
    #                                                         1000:1200,
    #                                              1000:1200],
    #                               1000,
    #                               1200, title_name='Embryonic Stem Cell ' \
    #                                                 'Chr15 '
    #                                                '1000-1200')
    # plot_scores_states_boundaries(scores_mes15[1000:1200],
    #                               predicted_states_mes15[
    #                               1000:1200],
    #                               boundaries_mes15[1000:1200],
    #                               hic_matrix_mes15[
    #                                                  1000:1200,
    #                                                 1000:1200],
    #                               1000,
    #                               1200, title_name='Mesendoderm Cell Chr15 '
    #                                                '1000-1200')
    # plot_scores_states_boundaries(scores_h1[600:800], predicted_states_h1[
    #                                                     600:800],
    #                               boundaries_h1[600:800], hic_matrix_h1[600:800,
    #                                              600:800],
    #                               600,
    #                               800, title_name='Embryonic Stem Cell ' \
    #                                                 'Chr15 '
    #                                                '600-800')
    # plot_scores_states_boundaries(scores_mes15[600:800],
    #                               predicted_states_mes15[
    #                               600:800],
    #                               boundaries_mes15[600:800],
    # hic_matrix_mes15[
    #                                                  600:800,
    #                                                 600:800],
    #                               600,
    #                               800, title_name='Mesendoderm Cell Chr15 '
    #                                                '600-800')
    # plot_scores_states_boundaries(scores_h1[800:1000], predicted_states_h1[
    #                                                     800:1000],
    #                               boundaries_h1[800:1000], hic_matrix_h1[800:1000,
    #                                              800:1000],
    #                               800,
    #                               1000, title_name='Embryonic Stem Cell ' \
    #                                                 'Chr15 '
    #                                                '800-1000')
    # plot_scores_states_boundaries(scores_mes15[800:1000],
    #                               predicted_states_mes15[
    #                               800:1000],
    #                               boundaries_mes15[800:1000],
    #                               hic_matrix_mes15[
    #                               800:1000,
    #                               800:1000],
    #                               800,
    #                               1000, title_name='Mesendoderm Cell Chr15 '
    #                                               '800-1000')
    #
    #
    # plot_scores_states_boundaries(scores_h1, predicted_states_h1,
    #                               boundaries_h1, hic_matrix_h1,
    #                               0,
    #                               2564, title_name='Embryonic Stem Cell ' \
    #                                                 'Chr15 '
    #                                                '0-2564')
    plot_scores_states_boundaries(scores_mes15,
                                  predicted_states_mes15,
                                  boundaries_mes15,
                                  hic_matrix_mes15,
                                  0,
                                  2564, title_name='Mesendoderm Cell Chr15 '
                                                  '0-2564')

    # plot_scores_states_boundaries(scores_lv13[2000:2200],
    #                               predicted_states_lv13[
    #                                                  2000:2200],
    #                               boundaries_lv13[2000:2200],
    #                               contact_matrix_lv13[
    #                                                           2000:3200,
    #                                              2000:2200],
    #                               2000,
    #                               2200, title_name='Left Ventricle Chr13 '
    #                                                '2000-2200')
    # plot_scores_states_boundaries(scores_rv13[2000:2200],
    #                               predicted_states_rv13[
    #                                                  2000:2200],
    #                               boundaries_rv13[2000:2200],
    #                               contact_matrix_rv13[
    #                                                           2000:2200,
    #                                              2000:2200],
    #                               2000,
    #                               2200, title_name='Right Ventricle Chr13 '
    #                                                '2000-2200')
    # plot_scores_states_boundaries(scores_lv14[2000:2200],
    #                               predicted_states_lv14[
    #                                                  2000:2200],
    #                               boundaries_lv14[2000:2200],
    #                               contact_matrix_lv14[
    #                                                           2000:2200,
    #                                              2000:2200],
    #                               2000,
    #                               2200, title_name='Left Ventricle Chr14 '
    #                                                '2000-2200')
    # plot_scores_states_boundaries(scores_rv14[2000:2200],
    #                               predicted_states_rv14[
    #                                                  2000:2200],
    #                               boundaries_rv14[2000:2200],
    #                               contact_matrix_rv14[
    #                                                           2000:2200,
    #                                              2000:2200],
    #                               2000,
    #                               2200, title_name='Right Ventricle Chr14 '
    #                                                '2000-2200')
    # plot_scores_states_boundaries(scores_lv13[1000:1200],
    #                               predicted_states_lv13[
    #                                                     1000:1200],
    #                               boundaries_lv13[1000:1200],
    #                               contact_matrix_lv13[1000:1200, 1000:1200],
    #                               1000,
    #                               1200, title_name='Left Ventricle Chr14 '
    #                                                '1000-1200')
    # plot_scores_states_boundaries(scores_rv13[1000:1200],
    #                               predicted_states_rv13[
    #                               1000:1200],
    #                               boundaries_rv13[1000:1200],
    #                               contact_matrix_rv13[1000:1200, 1000:1200],
    #                               1000,
    #                               1200, title_name='Right Ventricle Chr14 '
    #                                                '1000-1200')
    #
    # plot_scores_states_boundaries(scores_lv21[600:800],
    #                               predicted_states_lv21[
    #                                                     600:800],
    #                               boundaries_lv21[600:800],
    #                               contact_matrix_lv21[600:800, 600:800],
    #                               600,
    #                               800, title_name='Left Ventricle Chr21 '
    #                                                '600-800')
    # plot_scores_states_boundaries(scores_rv21[600:800],
    #                               predicted_states_rv21[
    #                               600:800],
    #                               boundaries_rv21[600:800],
    #                               contact_matrix_rv21[600:800, 600:800],
    #                               600,
    #                               800, title_name='Right Ventricle Chr21 '
    #                                                '600-800')
    # plot_scores_states_boundaries(scores_lv13[1600:1800],
    #                               predicted_states_lv13[
    #                                                     1600:1800],
    #                               boundaries_lv13[1600:1800],
    #                               contact_matrix_lv13[1600:1800, 1600:1800
    #                               ], 1600,
    #                               1800, title_name='Left Ventricle Chr13 '
    #                                                '1600-1800')
    # plot_scores_states_boundaries(scores_rv13[1600:1800],
    #                               predicted_states_rv13[
    #                                                     1600:1800],
    #                               boundaries_rv13[1600:1800],
    #                               contact_matrix_rv13[1600:1800, 1600:1800
    #                               ], 1600,
    #                               1800, title_name='Right Ventricle Chr13 '
    #                                                '1600-1800')
    # plot_scores_states_boundaries(scores_lv14[1600:1800],
    #                               predicted_states_lv14[
    #                                                     1600:1800],
    #                               boundaries_lv14[1600:1800],
    #                               contact_matrix_lv14[1600:1800, 1600:1800
    #                               ], 1600,
    #                               1800, title_name='Left Ventricle Chr14 '
    #                                                '1600-1800')
    # plot_scores_states_boundaries(scores_rv14[1600:1800],
    #                               predicted_states_rv14[
    #                                                     1600:1800],
    #                               boundaries_rv14[1600:1800],
    #                               contact_matrix_rv14[1600:1800, 1600:1800
    #                               ], 1600,
    #                               1800, title_name='Right Ventricle Chr14 '
    #                                                '1600-1800')
    #
    # plot_scores_states_boundaries(scores_lv21[400:600],
    #                               predicted_states_lv21[
    #                                                     400:600],
    #                               boundaries_lv21[400:600],
    #                               contact_matrix_lv21[400:600, 400:600
    #                               ], 400,
    #                               600, title_name='Left Ventricle Chr21 '
    #                                                '400-600')
    # plot_scores_states_boundaries(scores_rv21[400:600],
    #                               predicted_states_rv21[
    #                                                     400:600],
    #                               boundaries_rv21[400:600],
    #                               contact_matrix_rv21[400:600, 400:600
    #                               ], 400,
    #                               600, title_name='Right Ventricle Chr21 '
    #                                                '400-600')

    #
    plot_scores_comparison_graphs(scores_hc[3000:3100],
                                  insu_scores_hc[3000:3100],
                                  cell_name='HC Index 3000-3100',
                                  start_idx=3000,
                                  end_idx=3100)
    plot_scores_comparison_graphs(scores_hc[4000:4100],
                                  insu_scores_hc[4000:4100],
                                  cell_name='HC Index 4000-4100',
                                  start_idx=4000,
                                  end_idx=4100)
    plot_scores_comparison_graphs(scores_hc[300:400], insu_scores_hc[300:400],
                                  cell_name='HC Index 300-400', start_idx=300,
                                  end_idx=400)
    plot_scores_comparison_graphs(scores_hc[400:500], insu_scores_hc[400:500],
                                  cell_name='HC Index 400-500', start_idx=400,
                                  end_idx=500)
    # #
    # # plot_scores_comparison_graphs(scores[:100], scores_hc[0:100],
    # #                               cell_name='CO VS. HC Index 0-100',
    # #                               start_idx=0,
    # #                               end_idx=100)
    # # plot_scores_comparison_graphs(scores[100:200], scores_hc[100:200],
    # #                               cell_name='CO VS. HC Index 100-200',
    # #                               start_idx=100,
    # #                               end_idx=200)
    # plot_scores_comparison_graphs(scores[200:300], scores_hc[200:300],
    #                               cell_name='CO VS. HC Index 200-300',
    #                               start_idx=200,
    #                               end_idx=300)
    # plot_scores_comparison_graphs(scores[300:400], scores_hc[300:400],
    #                               cell_name='CO VS. HC Index 300-400',
    #                               start_idx=300,
    #                               end_idx=400)
    # #
    plot_scores_comparison_graphs(scores_tro[3000:3100], scores_mes[3000:3100],
                                  cell_name='Trophoblasts-Like Cell VS. '
                                            'Mesododerm Cell '
                                            '3000-3100',
                                  start_idx=3000,
                                  end_idx=3100)

    plot_scores_comparison_graphs(scores_mes15[1200:1300], scores_h1[
                                                           1200:1300],
                                  cell_name='Mesododerm VS Embryonic '
                                            'Stem Cell (CHR15)'
                                            '1200-1300',
                                  start_idx=1200,
                                  end_idx=1300)

    # # plot_scores_comparison_graphs(scores[:100], scores_li[0:100],
    # #                               cell_name='CO VS. LI Index 0-100',
    # #                               start_idx=0,
    # #                               end_idx=100)
    # # plot_scores_comparison_graphs(scores[100:200], scores_li[100:200],
    # #                               cell_name='CO VS. HC Index 100-200',
    # #                               start_idx=100,
    # #                               end_idx=200)
    # plot_scores_comparison_graphs(scores[200:300], scores_li[200:300],
    #                               cell_name='CO VS. LI Index 200-300',
    #                               start_idx=200,
    #                               end_idx=300)
    # plot_scores_comparison_graphs(scores[300:400], scores_li[300:400],
    #                               cell_name='CO VS. LI Index 300-400',
    #                               start_idx=300,
    #                               end_idx=400)
    #
    #
    # # boundaries = find_transition_indices(predicted_states[:200])
    #
    # # plot_scores_and_boundaries(scores[:200], boundaries)
    # #
    # # plot_scores_states_boundaries(scores[:200], predicted_states[:200],
    # #                               boundaries, 0,
    # #                               200)
    #
    # # plot_scores_states_boundaries(scores[200:400], predicted_states[200:400],
    # #                               boundaries2, 200,
    # #                               400)
    #
    # # predicted_states_insu, hmm_model_insu, scores_insu, means_insu, \
    # # covs_insu, \
    # # transition_points_insu = \
    # #     train_hmm_model(
    # #         np.array(
    # #             hic_matrix), compute_score, bin_size, square_size,
    # #         score_method_str='insulation_score')
    #
    # # plot_scores_graph(insulation_scores[:100])
    # # plot_scores_states(insulation_scores[:100], predicted_states[:100])
    # # plot_scores_graph(insulation_scores_weighted[:100])
    # # plot_scores_graph(insulation_scores_gradual[:100])
    #
    # # print("Insulation scores:", insulation_scores)
    #
    # # plot_scores_states(scores[:100], predicted_states[:100], 0, 100)
    # # plot_scores_states(scores_insu[:100], predicted_states_insu[:100])
    # # plot_scores_states(scores[100:200], predicted_states[100:200], 100, 200)
    # # plot_scores_states(scores[200:300], predicted_states[200:300], 200, 300)
    # # plot_scores_states(scores[300:400], predicted_states[300:400], 300, 400)
    # # plot_scores_states(scores[400:500], predicted_states[400:500], 400, 500)

    # pca_A_B_compartment(hic_matrix_co)
    # pca_A_B_compartment(hic_matrix_hc, title_name='Hippocampus')
    # pca_A_B_compartment(hic_matrix_li, title_name='Liver')
