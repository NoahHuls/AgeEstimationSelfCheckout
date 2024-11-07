using AgeEstimationSelfCheckout.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Text.Json;
using System.Text;

namespace AgeEstimationSelfCheckout.Pages
{
    public class ScanModel : PageModel
    {
        public static List<Product> Products { get; set; } = new List<Product>();
        public static bool AutomaticAgeVerification { get; set; } = false;
        public static bool AgeVerificationIsAltered { get; set; } = false;
        public static int Countdown { get; set; } = 15;
        public PredictionResponse Result { get; set; }

        public class PredictionResponse
        {
            public Data data { get; set; }
            public string message { get; set; }
        }

        public class Data
        {
            public int error { get; set; }
            public string error_message { get; set; }
            public bool over_25 { get; set; }
            public List<float> predictions { get; set; }
        }

        public async Task<IActionResult> OnPostToEmployee()
        {
            await ToEmployee();

            TempData["PredictionResult"] = JsonSerializer.Serialize(Result.data);

            return RedirectToPage("/Employee");
        }

        public async Task ToEmployee()
        {
            //TODO: Await 3 sec

            if (AutomaticAgeVerification)
            {
                using var client = new HttpClient();
                var content = new StringContent("{}", Encoding.UTF8, "application/json");

                try
                {
                    var response = await client.PostAsync("http://localhost:5000/api/predict", content);

                    if (response.IsSuccessStatusCode)
                    {
                        var responseData = await response.Content.ReadAsStringAsync();
                        Result = JsonSerializer.Deserialize<PredictionResponse>(responseData);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Request failed: " + ex.Message);
                }
            }
        }

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
            _ = CountdownAsync();
        }

        static async Task CountdownAsync()
        {
            while (Countdown > 0)
            {
                await Task.Delay(1000);
                Countdown--;
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