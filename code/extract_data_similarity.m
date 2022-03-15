
all_subjects = load_all_subjects('ExperimentalResults/');
all_sessions = [all_subjects.sessions_combined];
[all_r1, all_r2, all_theta, all_eu_distances, all_similarities, all_separators, all_pair_ids] = extract_results(all_sessions);

save_mat_kmj('data/all_data_combined/all_subjects.mat', 'all_subjects');
save_mat_kmj('data/all_data_combined/all_sessions.mat', 'all_sessions');
save_mat_kmj('data/all_data_combined/all_eu_distances.mat', 'all_eu_distances');
save_mat_kmj('data/all_data_combined/all_similarities.mat', 'all_similarities');
save_mat_kmj('data/all_data_combined/all_separators.mat', 'all_separators');
save_mat_kmj('data/all_data_combined/all_sessions.mat', 'all_sessions');
save_mat_kmj('data/all_data_combined/all_r1.mat', 'all_r1');
save_mat_kmj('data/all_data_combined/all_r2.mat', 'all_r2');
save_mat_kmj('data/all_data_combined/all_theta.mat', 'all_theta');
save_mat_kmj('data/all_data_combined/all_pair_ids.mat', 'all_pair_ids');

