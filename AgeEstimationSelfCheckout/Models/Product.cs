﻿namespace AgeEstimationSelfCheckout.Models
{
    public class Product
    {
        public string Name { get; set; }
        public decimal Price { get; set; }
        public bool IsAgeRestricted { get; set; }
    }
}
