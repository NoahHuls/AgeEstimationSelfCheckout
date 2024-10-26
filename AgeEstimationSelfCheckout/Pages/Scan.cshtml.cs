using AgeEstimationSelfCheckout.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Collections.Generic;

namespace AgeEstimationSelfCheckout.Pages
{
    public class ScanModel : PageModel
    {
        public static List<Product> Products { get; set; } = new List<Product>();
        public static bool AutomaticAgeVerification { get; set; } = false;
        public static bool AgeVerificationIsAltered { get; set; } = false;

        public void OnGet(string action)
        {
            if (action == "addBread")
            {
                AddBread();
            }
            else if (action == "addBeer")
            {
                AddBeer();
            }
            else if (action == "addCheese")
            {
                AddCheese();
            }
        }

        public void OnPostToggle(bool isAutomaticAgeVerificationEnabled)
        {
            AutomaticAgeVerification = !AutomaticAgeVerification;
            AgeVerificationIsAltered = true;
        }

        public void AddBread()
        {
            Products.Add(new Product { Name = "Brood", IsAgeRestricted = false, Price = 1.49M });
        }

        public void AddBeer()
        {
            Products.Add(new Product { Name = "Bier", IsAgeRestricted = true, Price = 3.49M });
        }

        public void AddCheese()
        {
            Products.Add(new Product { Name = "Kaas", IsAgeRestricted = false, Price = 2.99M });
        }
    }
}