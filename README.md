## RayleighChebyshevSamples

A collection of samples demonstrating the use of the RayleighChebyshev class to compute eigenpairs of Hermitian linear operators associated with some number of algebraically smallest eigenvalues.

The RayleighChebyshev class instances are templated with respect to three classes 

(a) A vector class
 
(b) A linear operator class that acts upon instances the of the vector class and whose eigensystem is to be determined.
 
(c) A randomize operator class that when applied to an instance of the vector class sets the vector elements to have random entries.

The primary sample, and the sample one should look at first, is DiagOpEigTest, a self-contained sample that computes the eigenpairs of a diagonal operator. Instances of the classes (a)-(c) that satisfy the template requirements are included in the DiagOpEigTest directory. 

Note: The convergence behavior of the RayleighChebyshev procedure is determined by the distribution of the eigenvalues of the linear operator. The convergence is not influenced by the structure of the operator so testing the procedure using a diagonal operator is just as useful as testing the procedure on a non-diagonal operator.

The other samples illustrate the use of RayleighChebyshev with pre-existing operator and vector classes. These examples were constructed by altering the DiagOpEigTest sample and replacing the classes it uses with the operator and vector classes associated with other types of linear operator and vector classes. 

Build instructions, either by command line or using CMake, are described in the test programs. 

Rayleigh-Chebyshev procedure reference: Christopher R. Anderson, "A Rayleigh-Chebyshev procedure for finding the smallest eigenvalues and associated eigenvectors of large sparse Hermitian matrices" Journal of Computational Physics, Volume 229 Issue 19, September, 2010.


### Prerequisites

C++17

### Versioning

Release : 0.0.1

### Date

Oct 11, 2024 

### Authors

Chris Anderson

### License

GPLv3  For a copy of the GNU General Public License see <http://www.gnu.org/licenses/>.

 







