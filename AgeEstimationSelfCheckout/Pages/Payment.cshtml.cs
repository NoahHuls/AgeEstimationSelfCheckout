using AgeEstimationSelfCheckout.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using System.Text.Json;

namespace AgeEstimationSelfCheckout.Pages
{
    public class PaymentModel : PageModel
    {
        public List<Product> Products { get; set; } = new List<Product>();
        public decimal TotalPrice { get; set; }


        public void OnGet()
        {
        }

        public void OnPost(string products, decimal totalPrice)
        {
            TotalPrice = totalPrice;
            Products = JsonSerializer.Deserialize<List<Product>>(products);
        }
    }
}
