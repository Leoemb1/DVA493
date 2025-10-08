clear; clc;

% Ladda tränad Q-tabell
load('Q_table.mat');

n_bins = size(Q, 1);
x_bounds = [-2.4, 2.4];
x_dot_bounds = [-5, 5];
theta_bounds = [-12*pi/180, 12*pi/180];
theta_dot_bounds = [-5, 5];
bounds = [x_bounds; x_dot_bounds; theta_bounds; theta_dot_bounds];

max_steps = 3000;  % för animation räcker kortare sekvens

% Initiera tillstånd
state = [0, 0, 0, 0];

% Pendelparametrar
cart_width = 0.4;
cart_height = 0.2;
pole_length = 1.0; % för visualisering
step_time = 0.02;
max_time = max_steps * step_time;

figure;
axis([-3 3 -1.5 1.5]);
grid on;
hold on;

for t = 1:max_steps
    clf;
    hold on;
    axis([-2.5 2.5 -1.5 1.5]);
    grid on;

    % Plotta vagnen
    cart_x = state(1);
    rectangle('Position', [cart_x - cart_width/2, -cart_height/2, cart_width, cart_height], ...
              'FaceColor', [0 0.5 1]);

    % Plotta pendeln
    theta = state(3);
    pole_x = cart_x + pole_length * sin(theta);
    pole_y = cart_height/2 - pole_length * cos(theta);

    line([cart_x, pole_x], [cart_height/2, pole_y], 'LineWidth', 4, 'Color', 'r');
    plot(pole_x, pole_y, 'ko', 'MarkerFaceColor', 'k');

    elapsed_time = t * step_time;
    title(sprintf('Tid: %.2f s', elapsed_time), 'FontSize', 14);

    drawnow;

    % Diskretisera
    s_idx = discretize_state(state, n_bins, bounds);

    % Välj handling
    [~, action] = max(Q(s_idx(1), s_idx(2), s_idx(3), s_idx(4), :));
    force = -10 * (action == 1) + 10 * (action == 2);

    % Simulera nästa tillstånd
    next_state = simulate(force, state(1), state(2), state(3), state(4));

    % Avsluta om pendeln faller
    if abs(next_state(1)) > 2.4 || abs(next_state(3)) > (12 * pi / 180)
        title(sprintf('Pendeln föll vid %.2f sek', t * step_time));
        break;
    end

    state = next_state;

    if elapsed_time >= 60
        title('✅ Pendeln klarade 1 minut!', 'FontSize', 14);
        break;
    end

    pause(step_time);  % Synka med verklig tid
end
