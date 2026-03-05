import {
    Block,
    Div,
    TasksKanban,
    Task,
    TaskData,
    TaskStatus,
    AddTaskPopup,
    ClientLocation,
    Tools,
    Listener
} from '@src/classes';

export class TasksColumn extends Div {

    private head: Div;
    private scrollContainer: Div;
    private tasksContainer: Div;

    private beautifiedStatus: string;

    private tasks: Task[] = [];

    private rect: DOMRect;
    private resizeListener: Listener;

    constructor(private status: TaskStatus, private kanban: TasksKanban, parent: Block) {

        super('tasks-column', parent);

        this.beautifyStatus();

        this.head = new Div('head', this).write(this.beautifiedStatus);

        if (['PREDEFINED', 'TODO', 'DONE'].includes(this.status))
            new Div('add', this.head).onNative('click', this.onAdd.bind(this));

        this.scrollContainer = new Div('scroll-container', this);
        this.tasksContainer = new Div('tasks-container', this.scrollContainer);

        this.resizeListener = ClientLocation.get().on('resize', this.recordRect.bind(this));
    }

    /*
    **
    **
    */
    private onAdd() : void {

        new AddTaskPopup(this.kanban.getTeamId(), this.status);
    }

    /*
    **
    **
    */
    public instanciateTask(data: TaskData) {

        const task = new Task(data, this);

        task.on('grabbed-task', this.onGrabbedTask.bind(this));

        //============================
        //INSERTING TASK AT BEST PLACE
        //============================

        const index = this.tasks.findIndex(task => task.getData().time <= data.time);

        //======
        //IN DOM
        //======

        const children = this.tasksContainer.element.children;

        if (index >= 0 && index < children.length)
            this.tasksContainer.element.insertBefore(task.element, children[index]);
        else
            this.tasksContainer.element.appendChild(task.element);
        
        //==============
        //IN TASKS ARRAY
        //==============

        if (index >= 0 && index < this.tasks.length)
            this.tasks.splice(index, 0, task);
        else
            this.tasks.push(task);
    }

    /*
    **
    **
    */
    private onGrabbedTask(task: Task) : void {
        
        this.emit('grabbed-task', task);
    }
    
    /*
    **
    **
    */
    private beautifyStatus() : void {

        this.beautifiedStatus = this.status
            .toLowerCase()
            .replace(/_/g, " ")
            .replace(/\b\w/g, char => char.toUpperCase());
    }

    /*
    **
    **
    */
    public getStatus() : TaskStatus {

        return this.status;
    }

    /*
    **
    **
    */
    public getBeautifiedStatus() : string {

        return this.beautifiedStatus;
    }

    /*
    **
    **
    */
    public getTasks() : Task[] {

        return this.tasks;
    }

    /*
    **
    **
    */
    public deleteTask(task: Task) : void {

        const index = this.tasks.findIndex(t => t.getData().id === task.getData().id);

        if (index !== -1)
            this.tasks.splice(index, 1);

        task.delete();
    }

    /*
    **
    **
    */
    public isPointerInside() : boolean {

        const pointer = ClientLocation.get().pointer;

        return (
            pointer.x >= this.rect.left &&
            pointer.x < this.rect.right &&
            pointer.y >= this.rect.top &&
            pointer.y < this.rect.bottom
        );
    }

    /*
    **
    **
    */
    public setPotentialDropStatus(status: -1 | 0 | 1) {
        
        this.setData('potential-drop-status', status);
    }

    /*
    **
    **
    */
    public recordRect() : void {

        this.rect = this.element.getBoundingClientRect();
    }
    
    /*
    **
    **
    */
    public filter(text: string | null, categoryId: number | null, accountId: number | null) : void {

        for (const task of this.tasks) {
            
            //====
            //TEXT
            //====

            if (text !== null) {

                text = text.toLowerCase();

                const containsText = task.getData().title.toLowerCase().includes(text) ||
                    task.getData().description.toLowerCase().includes(text) ||
                    task.getData().category_title.toLowerCase().includes(text) ||
                    task.getData().creator.toLowerCase().includes(text) ||
                    (task.getData().executor && task.getData().executor.toLowerCase().includes(text));

                if (!containsText) {
                    task.syncHide();
                    continue;
                }
            }

            //========
            //CATEGORY
            //========

            if (categoryId !== null) {
            
                const isSameCategory = task.getData().category_id === categoryId;
            
                if (!isSameCategory) {
                    task.syncHide();
                    continue;
                }
            }

            //=======
            //ACCOUNT
            //=======

            if (accountId !== null) {

                const accountFound = task.getData().creator_account_id === accountId ||
                    task.getData().executor_account_id === accountId;

                if (!accountFound) {
                    task.syncHide();
                    continue;
                }
            }
            
            task.syncDisplay();
        }
    }

    /*
    **
    **
    */
    public updateTasks() : void {

        for (const task of this.tasks)
            task.updateSinceValue();
    }

    /*
    **
    **
    */
    public onBeforeDelete() : void {

        this.resizeListener.off();
    }
}