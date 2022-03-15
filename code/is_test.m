function [testing] = is_test()

global is_test_kmj;
testing = isequal(is_test_kmj, 1);

end