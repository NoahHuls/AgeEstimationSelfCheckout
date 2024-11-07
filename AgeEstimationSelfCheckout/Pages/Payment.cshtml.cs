using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Dynamic;

namespace AgeEstimationSelfCheckout.Pages
{
    public class PaymentModel : PageModel
    {
        [BindProperty]
        public bool ScannedAlcohol { get; set; }
        [BindProperty]
        public decimal TotalPrice { get; set; }
        [BindProperty]
        public bool AutomaticAgeVerification { get; set; }

        public void OnGet()
        {

        }
    }
}
