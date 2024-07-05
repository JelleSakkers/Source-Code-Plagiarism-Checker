#include<stdio.h>
#include<math.h>

int checkPrime(int num)
{
    // 0, 1 and negative numbers are not prime
    if(num < 2){
        return 0;
    }
    else{   
    // A number n is not a prime, if it can be factored into two factors a & b:
    // n = a * b

    /*Now a and b can't be both greater than the square root of n, since
    then the product a * b would be greater than sqrt(n) * sqrt(n) = n.
    So in any factorization of n, at least one of the factors must be
    smaller than the square root of n, and if we can't find any factors
    less than or equal to the square root, n must be a prime.
    */
        for(int i = 2; i < sqrt(num); i++)
        {
            if(num % i == 0)
            {
                return 0;
            }
        }
    }
    // the number would be prime if we reach here
    return 1;
}

int main()
{
    
    for(int i=1; i <= 100; i++){
        if(checkPrime(i))
            printf("%d ",i);
    }
 
    return 0;
}
//Time Complexity: O(N√N)
//Space Complexity O(1)

// This method is obviously  faster as has better time complexity