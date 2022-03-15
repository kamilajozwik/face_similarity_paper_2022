
all_subjects_iso = load_all_subjects('ExperimentalResults/isotropicity/');
all_sessions_iso = [all_subjects_iso.sessions_combined];
[all_r1_iso, all_r2_iso, all_theta_iso, all_eu_distances_iso, all_similarities_iso, all_separators_iso, all_pair_ids_iso] = extract_results(all_sessions_iso);

save_mat_kmj('data/all_data_combined/all_subjects_iso.mat', 'all_subjects_iso');
save_mat_kmj('data/all_data_combined/all_sessions_iso.mat', 'all_sessions_iso');
save_mat_kmj('data/all_data_combined/all_eu_distances_iso.mat', 'all_eu_distances_iso');
save_mat_kmj('data/all_data_combined/all_similarities_iso.mat', 'all_similarities_iso');
save_mat_kmj('data/all_data_combined/all_separators_iso.mat', 'all_separators_iso');
save_mat_kmj('data/all_data_combined/all_sessions_iso.mat', 'all_sessions_iso');
save_mat_kmj('data/all_data_combined/all_r1_iso.mat', 'all_r1_iso');
save_mat_kmj('data/all_data_combined/all_r2_iso.mat', 'all_r2_iso');
save_mat_kmj('data/all_data_combined/all_theta_iso.mat', 'all_theta_iso');
save_mat_kmj('data/all_data_combined/all_pair_ids_iso.mat', 'all_pair_ids_iso');
