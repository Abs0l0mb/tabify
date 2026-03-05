import { 
    Api,
    Block,
    Div,
    View
} from '@src/classes';

export class ProgressReport extends Div {
    
    static readonly MAX_WORK_TIME = 160;

    private userCategoryUpdates: Record<number, Record<number, number>> = {};
    public globalView: View;
    private userViews: Record<number, View> = {};
    
    private usersDiv: Div;

    private promises: Promise<void>[] = [];

    private accountsData: any[];
    private allTasksByPeriodByTeam: any[];

    constructor(private teamId: number, private start: Date, private end: Date, parent: Block) {

        super('progress-report module', parent);
        
        this.render();
    }

    /*
    **
    **
    */
    private async render() : Promise<void> {

        this.accountsData = await Api.get('/progress-report/accounts');

        this.allTasksByPeriodByTeam = await Api.get('/progress-report/tasks', {
            teamId: this.teamId,
            start: this.start,
            end: this.end
        });

        const aggregatedTeamCategories = this.aggregateData(this.allTasksByPeriodByTeam);
        
        const globalViewContainer = new Div('global-view-container', this);

        console.log('hello');

        this.globalView = new View({
            label: 'Global Team View',
            data: aggregatedTeamCategories,
            editable: false,
            maxTime: this.accountsData.length * ProgressReport.MAX_WORK_TIME
        }, globalViewContainer);

        this.globalView.on('promise', promise => {
            this.promises.push(promise);
        });

        this.globalView.addClass('global');
        
        this.on('update', () => {
            this.updateGlobalAggregation();
        });

        this.usersDiv = new Div('users', this);

        this.accountsData.forEach(userAccount => {
            
            const userFullName = `${userAccount.first_name} ${userAccount.last_name}`;
            
            const userTasks = this.allTasksByPeriodByTeam.filter(task => task.executor_account_id === userAccount.account_id);

            const aggregatedUserCategories = this.aggregateData(userTasks);
            
            this.userCategoryUpdates[userAccount.account_id] = {};

            for (const catId in aggregatedUserCategories)
                this.userCategoryUpdates[userAccount.account_id][parseInt(catId, 10)] = aggregatedUserCategories[catId].totalTime;
            
            const userView = new View({
                label: userFullName,
                data: aggregatedUserCategories,
                editable: true,
                maxTime: ProgressReport.MAX_WORK_TIME
            }, this.usersDiv);
            
            userView.on('update', (updatedTimes: Record<number, number>) => {
                this.userCategoryUpdates[userAccount.account_id] = updatedTimes;
                this.updateGlobalAggregation();
            });
            
            this.userViews[userAccount.account_id] = userView;
        });
        
        this.updateGlobalAggregation();
    }
    
    
    /*
    **
    **
    */
    public aggregateData(taskList: any[]): Record<number, any> {

        const aggregatedCategories: Record<number, any> = {};

        taskList.forEach(task => {

            if (!aggregatedCategories[task.category_id]) {
                
                aggregatedCategories[task.category_id] = {
                    category_id: task.category_id,
                    category_title: task.category_title,
                    category_description: task.category_description,
                    tasks: [],
                    totalTime: 0
                };
            }

            aggregatedCategories[task.category_id].tasks.push(task);
            aggregatedCategories[task.category_id].totalTime += task.duree;
        });

        return aggregatedCategories;
    }
    
    /*
    **
    **
    */
    public updateGlobalAggregation(): void {

        const globalAggregates: Record<number, any> = {};
        const staticAggregation = this.aggregateData(this.allTasksByPeriodByTeam);

        for (const catId in staticAggregation) {
            const numericCatId = parseInt(catId, 10);
            globalAggregates[numericCatId] = {
                category_id: numericCatId,
                category_title: staticAggregation[numericCatId].category_title,
                category_description: staticAggregation[numericCatId].category_description,
                totalTime: 0
            };
        }
        
        for (const userId in this.userCategoryUpdates) {
            const userCategoryTimes = this.userCategoryUpdates[userId];
            for (const catId in userCategoryTimes) {
                const numericCatId = parseInt(catId, 10);
                if (!globalAggregates[numericCatId]) {
                    globalAggregates[numericCatId] = {
                        category_id: numericCatId,
                        category_title: "Unknown",
                        category_description: "",
                        totalTime: 0
                    };
                }
                globalAggregates[numericCatId].totalTime += userCategoryTimes[catId];
            }
        }
        
        for (const catId in globalAggregates) {
            const numericCatId = parseInt(catId, 10);
            this.globalView.updateCategoryTime(numericCatId, globalAggregates[numericCatId].totalTime);
        }
    }
    
    /*
    **
    **
    */
    public generateExportHTML(): string {
        
        const globalAggregates: Record<number, any> = {};
        const staticAggregation = this.aggregateData(this.allTasksByPeriodByTeam);
        
        for (const catId in staticAggregation) {

            const catNum = parseInt(catId, 10);

            globalAggregates[catNum] = {
                category_title: staticAggregation[catNum].category_title,
                category_description: staticAggregation[catNum].category_description,
                totalTime: 0,
                tasks: staticAggregation[catNum].tasks
            };
        }
        
        for (const userId in this.userCategoryUpdates) {

            const userCatTimes = this.userCategoryUpdates[userId];

            for (const catId in userCatTimes) {

                const catNum = parseInt(catId, 10);

                if (!globalAggregates[catNum]) {

                    globalAggregates[catNum] = {
                        category_title: "Unknown",
                        category_description: "",
                        totalTime: 0,
                        tasks: []
                    };
                }

                globalAggregates[catNum].totalTime += userCatTimes[catId];
            }
        }
        
        let allocatedGlobal = 0;
        for (const catId in globalAggregates)
            allocatedGlobal += globalAggregates[catId].totalTime;
        
        const categoryListItems = Object.keys(globalAggregates).map(catId => {
            const cat = globalAggregates[parseInt(catId, 10)];
            const pct = allocatedGlobal > 0 ? Math.round((cat.totalTime / allocatedGlobal) * 100) : 0;
            return `<li>${cat.category_title} : ${pct}%</li>`;
        }).join("\n");
        
        const detailSections = Object.keys(globalAggregates).map(catId => {

            const cat = globalAggregates[parseInt(catId, 10)];
            const pct = allocatedGlobal > 0 ? Math.round((cat.totalTime / allocatedGlobal) * 100) : 0;
            
            const taskRows = cat.tasks.map((task: any) => {
                return `<tr><td>${task.task_title}</td><td>${task.task_description}</td></tr>`;
            }).join("\n");

            return `
                <div class="category-detail">
                    <h3>${cat.category_title} : ${pct}%</h3>
                    <p>${cat.category_description}</p>
                    <table border="1" cellspacing="0" cellpadding="4">
                    <thead>
                        <tr><th>Task Title</th><th>Task Description</th></tr>
                    </thead>
                    <tbody>
                        ${taskRows}
                    </tbody>
                    </table>
                </div>
            `;

        }).join("\n");
        
        let chartDataURL = "";

        if (this.globalView && this.globalView.chart && typeof this.globalView.chart.getDataURL === "function")
            chartDataURL = this.globalView.chart.getDataURL();
            
        return `
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                body { font-family: 'Roboto', sans-serif; margin: 20px; }
                header { text-align: center; margin-bottom: 40px; }
                h1 { font-size: 2.5em; margin-bottom: 0.2em; }
                h2 { font-size: 1.8em; margin-top: 40px; }
                h3 { font-size: 1.4em; margin-top: 20px; }
                .resume-table { border: 0px; border-collapse: collapse; }
                .resume-table td, .resume-table th { border: none; }
                footer { position: fixed; bottom: 10px; width: 100%; text-align: center; font-size: 0.8em; color: #888; }
                table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
                .category-detail { margin-bottom: 30px; }
                </style>
            </head>
            <body>
                <header>
                <h1>Progress Report</h1>
                </header>
                <table class="resume-table">
                <tr>
                    <td class="left-column">
                    <h2>Résumé</h2>
                    <ul>
                        ${categoryListItems}
                    </ul>
                    </td>
                    <td class="right-column">
                    <h2>Graphique</h2>
                    ${chartDataURL ? `<img src="${chartDataURL}" alt="Global Chart" style="max-width: 100%;">` : '<p>Graphique non disponible</p>'}
                    </td>
                </tr>
                </table>
                <section class="details">
                <h2>Détail</h2>
                ${detailSections}
                </section>
                <footer>
                Provided by JPSI
                </footer>
            </body>
            </html>
        `;
    }
}