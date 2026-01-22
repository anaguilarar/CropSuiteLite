document$.subscribe(function() {
    // Select the table with the specific class
    var table = $('.datatable');
    
    // Check if the table exists and isn't already a DataTable
    if (table.length && !$.fn.DataTable.isDataTable(table)) {
        table.DataTable({
            responsive: true,
            pageLength: 10,
            order: [] // Disable initial sort if needed
        });
    }
});