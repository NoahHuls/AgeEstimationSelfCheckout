using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Dynamic;
using System.Text;

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

        public async Task OnPostAsync()
        {
            using var client = new HttpClient();
            var content = new StringContent("{}", Encoding.UTF8, "application/json");

            try
            {
                var response = await client.PostAsync("http://localhost:5000/api/predict", content);

                if (response.IsSuccessStatusCode)
                {
                    var responseData = await response.Content.ReadAsStringAsync();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Request failed: " + ex.Message);
            }
        }
    }
}
