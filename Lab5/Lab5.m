clear; clc;

%% --- Q-learning parametrar ---
alpha = 0.2;             % Inlärningshastighet
gamma = 0.99;            % Diskonteringsfaktor
epsilon = 1.0;           % Utforskningsgrad (epsilon-greedy)
epsilon_decay = 0.995;   % Hur snabbt vi minskar utforskning
epsilon_min = 0.05;      % Minsta tillåtna utforskning

n_episodes = 5000;       % Antal träningsavsnitt
max_steps = 3000;        % Max steg per avsnitt (60 sekunder)

n_bins = 15;             % Antal diskretiseringsnivåer per tillståndsvariabel
Q = zeros(n_bins, n_bins, n_bins, n_bins, 2);  % Q-tabell (4D tillstånd + 2 actions)

%% --- Tillståndsgränser (justerade) ---
x_bounds = [-2.4, 2.4];
x_dot_bounds = [-5, 5];                     % Ökat från [-3, 3]
theta_bounds = [-12*pi/180, 12*pi/180];
theta_dot_bounds = [-5, 5];                % Ökat från [-2, 2]

bounds = [x_bounds; x_dot_bounds; theta_bounds; theta_dot_bounds];

%% --- Loggning för visualisering ---
reward_per_episode = zeros(1, n_episodes);   % Sparar reward för varje episod

%% --- Träningsloop ---
for episode = 1:n_episodes
    % Initiera starttillstånd
    state = [0, 0, 0, 0];   % (x, x_dot, theta, theta_dot)
    total_reward = 0;

    for t = 1:max_steps
        % Diskretisera tillståndet
        s_idx = discretize_state(state, n_bins, bounds);

        % Epsilon-greedy actionval
        if rand < epsilon
            action = randi([1, 2]); % 1 = vänster, 2 = höger
        else
            [~, action] = max(Q(s_idx(1), s_idx(2), s_idx(3), s_idx(4), :));
        end

        % Översätt action till kraft
        force = -10 * (action == 1) + 10 * (action == 2);

        % Simulera nästa tillstånd
        next_state = simulate(force, state(1), state(2), state(3), state(4));

        % Kontrollera om pendeln faller eller vagnen kör utanför
        done = any(isnan(next_state)) || abs(next_state(1)) > 2.4 || abs(next_state(3)) > (12 * pi / 180);

        % Reward: +1 för varje steg överlevt, -10 om fall
        if done
            reward = -10;
            fprintf('Pendeln föll vid steg %d, x=%.3f, vinkel θ=%.3f grader\n', ...
                t, next_state(1), next_state(3)*180/pi);
        else
            reward = 1;
        end

        % Diskretisera nästa tillstånd (endast om inte done)
        if ~done
            next_idx = discretize_state(next_state, n_bins, bounds);
        else
            next_idx = s_idx; % Om done, håll kvar nuvarande index för uppdatering
        end

        % Q-värdeuppdatering (Bellman)
        best_next_q = max(Q(next_idx(1), next_idx(2), next_idx(3), next_idx(4), :));
        Q(s_idx(1), s_idx(2), s_idx(3), s_idx(4), action) = ...
            Q(s_idx(1), s_idx(2), s_idx(3), s_idx(4), action) + ...
            alpha * (reward + gamma * best_next_q - Q(s_idx(1), s_idx(2), s_idx(3), s_idx(4), action));

        % Gå vidare
        state = next_state;
        total_reward = total_reward + reward;

        if done
            break;
        end
    end

    % Logga reward (antal steg överlevt, utan straff)
    reward_per_episode(episode) = max(total_reward, 0);

    % Minska epsilon gradvis
    epsilon = max(epsilon * epsilon_decay, epsilon_min);

    % Utskrift för var 50:e episod
    if mod(episode, 50) == 0
        fprintf('Episod %d: Steg = %d, Reward = %d, Epsilon = %.3f\n', episode, t, total_reward, epsilon);
    end

    % Early stopping om agenten klarar 60 sek (3000 steg)
    if total_reward >= 3000
        fprintf('✅ Agenten klarade 60 sekunder på episod %d!\n', episode);
        % Spara Q-tabellen
        save('Q_table.mat', 'Q');
        break;
    end
end

%% --- Efter träning ---
% Om inte early stopping inträffade, spara ändå Q-tabellen
if total_reward < 3000
    save('Q_table.mat', 'Q');
end

% Plotta reward-utveckling
figure;
plot(reward_per_episode);
xlabel('Episod');
ylabel('Reward (antal steg överlevt)');
title('Reward per episod (Träningsförlopp)');
grid on;
