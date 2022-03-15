function [ all_subjects, all_sessions ] = load_all_subjects(directory)

%kmj session_1 = [];
%kmj session_2 = [];
all_sessions = [];
all_subjects_struct = struct();
all_subjects = [];

files = dir([directory '/*.mat']);

% n = 10;
n = length(files);

if n == 0
    error(['No files found in ' directory]);
end

% Looping a first time create the all_subjects_struct
for i=1:n
    filename = files(i).name;
    subject_name = char(regexp(filename, '^[a-zA-Z]+', 'match'));
    all_subjects_struct.(subject_name).sessions = [];
end

% Filling the data
for i=1:n
    filename = files(i).name
    subject_name = char(regexp(filename, '^[a-zA-Z]+', 'match'));
    load([directory '/' filename]);
    %kmj all_subjects_struct.(subject_name).session_1 = [all_subjects_struct.(subject_name).session_1; resultsStruct];
    %kmj all_subjects_struct.(subject_name).session_2 = [all_subjects_struct.(subject_name).session_1; resultsStruct];
    all_subjects_struct.(subject_name).sessions = [all_subjects_struct.(subject_name).sessions; resultsStruct];
    all_sessions = [all_sessions, resultsStruct];
end

subject_names = fieldnames(all_subjects_struct);
for i = 1:numel(subject_names)
   subject = all_subjects_struct.(subject_names{i});
   subject.name = subject_names{i};
   subject.sessions_combined = subject.sessions(:)';
   all_subjects = [all_subjects, subject];
end

end